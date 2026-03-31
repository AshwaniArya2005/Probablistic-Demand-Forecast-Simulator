"""Phase 3: clean M5, derive closed-day rule + velocity segments from data <= tuning cutoff, draw the 100-item subset,
write data/processed/panel.parquet (long format, one row per series-day on sale). Run: uv run python ml/prepare.py"""
import numpy as np
import pandas as pd

from config import *

cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
prices = pd.read_csv(RAW / "sell_prices.csv")
prices = prices[prices.store_id.isin(STORES)]
sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
sales = sales[sales.store_id.isin(STORES)]
d_cols = [c for c in sales if c.startswith("d_")]
date_of = cal.set_index("d").date
train_days = [d for d in d_cols if date_of[d] <= TUNING_CUTOFF]

# --- closed-day rule: month-days where every store closed in every training-year occurrence ---
tot = sales.groupby("store_id")[d_cols].sum().T.set_index(date_of[d_cols].values)
train_tot = tot[tot.index <= TUNING_CUTOFF]
closed_sd = train_tot < 0.01 * train_tot.median()          # near-zero: < 1% of the store's median daily total
md = train_tot.index.strftime("%m-%d")
by_md = closed_sd.groupby(md).mean().min(axis=1)           # share of training years closed, worst store
n_years = pd.Series(md).value_counts()
CLOSED_MD = sorted(m for m, share in by_md.items() if share == 1 and n_years[m] >= 3)
stray = closed_sd[~np.isin(md, CLOSED_MD)].stack()
print(f"closed-day calendar rule (all 3 stores closed in every training year): {CLOSED_MD}")
print(f"training store-days below threshold outside the rule: {int(stray.sum())}")
print("max store total on rule days, training years:", int(train_tot[np.isin(md, CLOSED_MD)].max().max()),
      "| min on other days:", int(train_tot[~np.isin(md, CLOSED_MD)].min().min()))
cal["closed"] = cal.date.dt.strftime("%m-%d").isin(CLOSED_MD)

# --- first on-sale day per series ---
wk_start = cal.groupby("wm_yr_wk").date.min()
first = prices.groupby(["store_id", "item_id"]).wm_yr_wk.min().map(wk_start).rename("first_date").reset_index()
sales = sales.merge(first, on=["store_id", "item_id"])

# --- velocity from data <= cutoff: mean daily sales since first on-sale day, closed days excluded ---
is_closed = cal.set_index("d").closed
open_train = [d for d in train_days if not is_closed[d]]
days = pd.DataFrame({"d": open_train, "date": date_of[open_train].values})
mask = (days.date.values[None, :] >= sales.first_date.values[:, None])        # series x day: on sale yet
sales["velocity"] = (sales[open_train].to_numpy() * mask).sum(1) / mask.sum(1)

# --- eligible pool: item on sale >= 2 years before the cutoff in all 3 stores ---
sales["hist_ok"] = sales.first_date <= TUNING_CUTOFF - pd.Timedelta(days=MIN_HISTORY_DAYS)
elig = sales.groupby("item_id").hist_ok.all()
pool = sales[sales.item_id.map(elig)].copy()
print(f"eligible items: {int(elig.sum())} of {len(elig)}; series {len(pool)}")

lo, hi = pool.velocity.quantile([1 / 3, 2 / 3])
pool["segment"] = pd.cut(pool.velocity, [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
print(f"series velocity tercile cuts (units/day): {lo:.3f}, {hi:.3f}")

# --- stratified item sample: cells = dept x item-velocity tercile; min 1 per cell, rest proportional (largest remainder) ---
items = pool.groupby("item_id").agg(dept_id=("dept_id", "first"), velocity=("velocity", "mean"))
ilo, ihi = items.velocity.quantile([1 / 3, 2 / 3])
items["stratum"] = pd.cut(items.velocity, [-np.inf, ilo, ihi, np.inf], labels=["low", "mid", "high"])
cells = items.groupby(["dept_id", "stratum"], observed=True).size()
extra = (N_ITEMS - len(cells)) * cells / cells.sum()
alloc = 1 + extra.astype(int)
alloc += (extra - extra.astype(int)).rank(ascending=False, method="first").le(N_ITEMS - alloc.sum()).astype(int)
rng = np.random.default_rng(SEED)
chosen = sorted(i for (dept, s), n in alloc.items()
                for i in rng.choice(sorted(items.index[(items.dept_id == dept) & (items.stratum == s)]), n, replace=False))
assert len(chosen) == N_ITEMS
print("sampled items by stratum:", items.loc[chosen].stratum.value_counts().to_dict())
print("sampled items by dept:", items.loc[chosen].dept_id.value_counts().to_dict())

# --- long panel: rows before first sell_price dropped (not yet on sale, not zero demand) ---
sub = pool[pool.item_id.isin(chosen)]
long = sub.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "first_date", "segment"],
                value_vars=d_cols, var_name="d", value_name="sales")
long = long.merge(cal, on="d")
n_all = len(long)
long = long[long.date >= long.first_date]
print(f"panel rows: {n_all} -> {len(long)} after dropping {n_all - len(long)} pre-launch rows")
long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
long["snap"] = np.select([long.state_id == s for s in ("CA", "TX", "WI")], [long.snap_CA, long.snap_TX, long.snap_WI])
print(f"on-sale days with no sell_price row (price gap): {int(long.sell_price.isna().sum())}")
print(f"on-sale days with sales > 0 but no price: {int(((long.sales > 0) & long.sell_price.isna()).sum())}")

keep = ["id", "item_id", "store_id", "dept_id", "cat_id", "segment", "date", "wm_yr_wk", "sales", "sell_price", "closed", "snap",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
panel = long[keep].sort_values(["id", "date"]).reset_index(drop=True)
panel = panel.astype({"sales": "int16", "sell_price": "float32", "snap": "int8", "segment": "category"})
PROCESSED.mkdir(parents=True, exist_ok=True)
panel.to_parquet(PROCESSED / "panel.parquet")
print(f"wrote {PROCESSED / 'panel.parquet'}: {len(panel)} rows, {panel.id.nunique()} series")
