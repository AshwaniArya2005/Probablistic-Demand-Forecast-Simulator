"""Phase 2: load the raw M5 files and check them against docs/design.md. Run: uv run python ml/inspect_data.py"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
flags = []


def check(label, got, want):
    ok = got == want
    shown = got if len(str(got)) < 120 else "(long value, matches)" if ok else "(long value)"
    print(f"[{'ok' if ok else 'DIFF'}] {label}: {shown}" + ("" if ok else f"  (design/M5 docs: {want})"))
    if not ok:
        flags.append(label)


# --- load ---
head = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns
id_cols = [c for c in head if not c.startswith("d_")]
d_cols = [c for c in head if c.startswith("d_")]
sales = pd.read_csv(RAW / "sales_train_evaluation.csv", dtype={c: "int16" for c in d_cols})
cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
prices = pd.read_csv(RAW / "sell_prices.csv")

# --- columns ---
print("== columns ==")
check("sales id cols", id_cols, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
check("sales day cols", (d_cols[0], d_cols[-1], len(d_cols)), ("d_1", "d_1941", 1941))
check("d_ contiguous", d_cols, [f"d_{i}" for i in range(1, 1942)])
check("calendar cols", list(cal.columns), ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
      "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])
check("prices cols", list(prices.columns), ["store_id", "item_id", "wm_yr_wk", "sell_price"])

# --- date range ---
print("== dates ==")
last_sales_date = cal.loc[cal.d == "d_1941", "date"].iloc[0]
check("first date", str(cal.date.min().date()), "2011-01-29")
check("d_1941 date (end of sales)", str(last_sales_date.date()), "2016-05-22")
print(f"[info] calendar runs to {cal.date.max().date()} ({len(cal)} days); days after d_1941 have no sales (M5 evaluation horizon)")
check("calendar dates unique + daily", cal.date.diff().dropna().eq(pd.Timedelta(days=1)).all(), True)

# --- counts ---
print("== counts ==")
check("series", len(sales), 30490)
check("items", sales.item_id.nunique(), 3049)
check("stores", sales.store_id.nunique(), 10)
check("states / depts / cats", (sales.state_id.nunique(), sales.dept_id.nunique(), sales.cat_id.nunique()), (3, 7, 3))
ca = sales[sales.store_id.isin(["CA_1", "CA_2", "CA_3"])]
print(f"[info] stores in CA: {sorted(sales[sales.state_id == 'CA'].store_id.unique())}")
check("CA_1..3 series", len(ca), 3049 * 3)
check("CA_1..3 items", ca.item_id.nunique(), 3049)
print(f"[info] CA_1..3 items by dept: {ca.drop_duplicates('item_id').dept_id.value_counts().to_dict()}")

# --- quality ---
print("== quality ==")
X = sales[d_cols].to_numpy()
check("sales nulls / negatives", (int(sales.isna().sum().sum()), int((X < 0).sum())), (0, 0))
check("prices nulls", int(prices.isna().sum().sum()), 0)
check("prices rows unique on (store,item,week)", not prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any(), True)
missing = set(zip(sales.store_id, sales.item_id)) - set(zip(prices.store_id, prices.item_id))
check("series with no price rows at all", len(missing), 0)
print(f"[info] all-zero series: {int((X.sum(axis=1) == 0).sum())}")

# --- design-doc rules ---
print("== design-doc rules ==")
test_start = last_sales_date - pd.Timedelta(weeks=26) + pd.Timedelta(days=1)
print(f"[info] test_start = {test_start.date()} (design: ~22 Nov 2015), tuning cutoff = {(test_start - pd.Timedelta(weeks=12)).date()}")
wk_start = cal.groupby("wm_yr_wk").date.min()
first_wk = prices.groupby(["store_id", "item_id"]).wm_yr_wk.min().map(wk_start).reset_index(name="first_price_date")
ca_first = first_wk[first_wk.store_id.isin(["CA_1", "CA_2", "CA_3"])]
for name, ref in [("test window", test_start), ("tuning cutoff", test_start - pd.Timedelta(weeks=12))]:
    ok = ca_first.first_price_date <= ref - pd.Timedelta(days=2 * 365)
    print(f"[info] CA_1..3 series on sale >= 2y before the {name}: {int(ok.sum())} of {len(ca_first)}; "
          f"items with all 3 stores: {int(ok.groupby(ca_first.item_id).all().sum())}")
xmas = cal[cal.date.dt.strftime("%m-%d") == "12-25"].d.tolist()
xmas = [d for d in xmas if d in d_cols]
print(f"[info] Dec 25 total CA_1..3 sales by year (closed-store check, Phase 3): "
      f"{dict(zip(cal.set_index('d').loc[xmas].year, ca[xmas].sum().tolist()))}")

print("\n== summary ==")
print("all as designed" if not flags else f"DIFFERS from design/M5 docs: {flags}")
