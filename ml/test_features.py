"""Leakage and correctness checks for ml/features.py. Run: uv run python ml/test_features.py (asserts; prints ok).
1. brute-force reference: every feature at random (series, origin) pairs recomputed with plain loops from panel.parquet
2. perturbation: scramble sales / prices after an origin t0 and rebuild; nothing observable at t <= t0 may change,
   except the targets and future-window price columns whose windows reach past t0 (and those must change, or the test is blind)."""
import numpy as np
import pandas as pd

from config import *
from features import *

rng = np.random.default_rng(0)
sales, price, cal, static = load()
F = build(sales, price, cal, static)
panel = pd.read_parquet(PROCESSED / "panel.parquet")

# ---------- 0. structure ----------
allowed = {c for P in PS for c in columns(P)}
assert not any(c.startswith("y_") for c in allowed), "target in model inputs"
assert allowed <= set(F.columns)
for P in PS:
    assert not set(columns(P, future_price=False)) & {f"price_mean_rel_p{P}", f"price_min_rel_p{P}"}
    assert set(columns(P)) - set(columns(P, future_price=False)) == {f"price_mean_rel_p{P}", f"price_min_rel_p{P}"}
assert "segment" not in F.columns, "segment labels use data after early origins: reporting/pooling only, never a feature"
assert set(cal.index[cal.n_closed == 1].strftime("%m-%d")) == {"12-25"}, "closed flag must be the Dec 25 calendar rule"
assert (F.age_days >= MIN_HISTORY - 1).all()
print("ok structure")

# ---------- 1. brute-force reference ----------
ser = {i: sales[i] for i in sales.columns}
pri = {i: price[i] for i in price.columns}
day = panel.groupby("date").agg(t1=("event_type_1", "first"), t2=("event_type_2", "first"), snap=("snap", "first"), closed=("closed", "first"))


def ref(sid, t, P):
    s, p = ser[sid], pri[sid]
    d = lambda k: t + pd.Timedelta(days=k)
    r = {f"lag_{k}": s.get(d(-k)) for k in LAGS}
    for w in WINDOWS:
        v = np.array([s[d(-k)] for k in range(w) if not day.closed[d(-k)] and not np.isnan(s[d(-k)])])
        r[f"mean_{w}"], r[f"std_{w}"] = v.mean(), v.std(ddof=1)
    for w in (28, 91):
        v = np.array([s[d(-k)] for k in range(w) if not day.closed[d(-k)] and not np.isnan(s[d(-k)])])
        r[f"zero_frac_{w}"] = (v == 0).mean()
    first = p.first_valid_index()
    r["age_days"] = (t - first).days
    sold = s[(s > 0) & (s.index <= t)]
    r["days_since_sale"] = (t - sold.index[-1]).days if len(sold) else r["age_days"]
    run = best = 0
    for k in range(ZR_WINDOW - 1, -1, -1):                        # oldest to newest
        if day.closed[d(-k)]:
            continue
        run = run + 1 if s[d(-k)] == 0 else 0
        best = max(best, run)
    r["zero_run_91"] = min(best, ZR_CAP)
    r[f"sum_last_{P}"] = sum(s[d(-k)] for k in range(P))
    r[f"sum_364_{P}"] = sum(s.get(d(-364 + k), np.nan) for k in range(1, P + 1))
    r["price"] = p[t]
    med = np.nanmedian([p.get(d(-k), np.nan) for k in range(MEDIAN_WINDOW)])
    r["price_rel_now"] = p[t] / med
    r["origin_dow"] = t.dayofweek
    fut = [d(k) for k in range(1, P + 1)]
    r[f"n_snap_p{P}"] = sum(day.snap[x] for x in fut)
    r[f"n_event_p{P}"] = sum(pd.notna(day.t1[x]) or pd.notna(day.t2[x]) for x in fut)
    for ty in EVENT_TYPES:
        r[f"n_{ty.lower()}_p{P}"] = sum(day.t1[x] == ty or day.t2[x] == ty for x in fut)
    r[f"n_weekend_p{P}"] = sum(x.dayofweek >= 5 for x in fut)
    r[f"n_closed_p{P}"] = sum(bool(day.closed[x]) for x in fut)
    r[f"price_mean_rel_p{P}"] = np.mean([p[x] for x in fut]) / med
    r[f"price_min_rel_p{P}"] = np.min([p[x] for x in fut]) / med
    r[f"y_p{P}"] = sum(s[x] for x in fut)
    return r


sample = F[F.date <= LAST_SALES_DATE - pd.Timedelta(days=14)]
picks = list(sample.sample(60, random_state=1).index)
picks += list(sample[(sample.date.dt.strftime("%m-%d").between("12-20", "12-30"))].sample(10, random_state=2).index)   # closed-day windows
picks += list(sample[sample.id == "FOODS_2_101_CA_2_evaluation"].sample(5, random_state=3).index)                      # 1,578-day leading zero run
picks += list(sample[sample.date == TEST_START].sample(10, random_state=4).index)
n = 0
for ix in picks:
    row = F.loc[ix]
    for P in PS:
        for k, v in ref(row.id, row.date, P).items():
            got = row[k]
            assert np.isclose(got, v, rtol=1e-4, atol=1e-4, equal_nan=True), (row.id, row.date.date(), P, k, got, v)
            n += 1
print(f"ok brute-force reference: {len(picks)} rows x 3 horizons, {n} values match")

# ---------- 2. perturbation ----------
t0 = pd.Timestamp("2014-06-15")
hist_cols = [c for c in F.columns if c not in ("id", "date") and not c.startswith("y_")]
pfut = [c for c in F.columns if c.startswith("price_mean_rel_p") or c.startswith("price_min_rel_p")]


def numeric(df, cols):
    return df[[c for c in cols if str(df[c].dtype) != "category"]]


def compare(F2, cols, mask):
    a, b = F[mask][["id", "date", *cols]], F2[mask][["id", "date", *cols]]
    assert a[["id", "date"]].equals(b[["id", "date"]]), "row set changed"
    return numeric(a, cols).fillna(-9e9).to_numpy(), numeric(b, cols).fillna(-9e9).to_numpy()


def scramble(frame, kind):
    m = frame.index > t0
    x = frame.copy()
    vals = rng.integers(0, 10, x[m].shape) if kind == "sales" else frame[m].to_numpy() * rng.uniform(0.5, 1.5, x[m].shape)
    x.loc[m] = np.where(frame[m].notna(), vals.astype("float32"), np.nan)
    return x


past = F.date <= t0
near = (F.date > t0 - pd.Timedelta(days=14)) & past

# 2a. scramble sales after t0
F2 = build(scramble(sales, "sales"), price, cal, static)
a, b = compare(F2, hist_cols, past)
assert np.array_equal(a, b), "a feature at t <= t0 changed when sales after t0 were scrambled"
ycols = [c for c in F.columns if c.startswith("y_")]
a, b = compare(F2, ycols, F.date <= t0 - pd.Timedelta(days=14))
assert np.array_equal(a, b), "targets with windows entirely <= t0 changed"
a, b = compare(F2, ycols, near)
assert not np.array_equal(a, b), "test is blind: targets reaching past t0 should have changed"
a, b = compare(F2, hist_cols, F.date > t0 + pd.Timedelta(days=30))
assert not np.array_equal(a, b), "test is blind: history after t0 should have changed"
print("ok perturbation 1: scrambling sales after t0 leaves every feature at t <= t0 unchanged")

# 2b. scramble prices after t0
F3 = build(sales, scramble(price, "price"), cal, static)
non_price_fut = [c for c in hist_cols if c not in pfut]
a, b = compare(F3, non_price_fut, past)
assert np.array_equal(a, b), "a non-future-price feature at t <= t0 changed when prices after t0 were scrambled"
a, b = compare(F3, pfut, F.date <= t0 - pd.Timedelta(days=14))
assert np.array_equal(a, b), "future-price windows entirely <= t0 changed"
a, b = compare(F3, pfut, near)
assert not np.array_equal(a, b), "test is blind: future-price windows reaching past t0 should have changed"
a, b = compare(F3, ycols, past)
assert np.array_equal(a, b), "prices must not change targets"
print("ok perturbation 2: scrambling prices after t0 changes only the future-price window columns that reach past t0")
print("all feature checks passed")
