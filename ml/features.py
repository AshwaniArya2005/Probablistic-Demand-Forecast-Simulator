"""Phase 4: features (and targets) as of origin t, one row per series x day. Definitions and leakage notes: docs/features.md.
Run: uv run python ml/features.py  -> data/processed/features.parquet (git-ignored). ml/test_features.py checks the leakage rules."""
import numpy as np
import pandas as pd

from config import *

PS = (7, 10, 14)                       # protection intervals, days
LAGS = range(7)                        # lag_k = sales on day t-k
WINDOWS = (7, 14, 28, 91)
MEDIAN_WINDOW, MEDIAN_MIN = 182, 91    # trailing median price ("usual price"), days
ZR_WINDOW, ZR_CAP = 91, 56             # longest zero run in the last 91 days, capped
MIN_HISTORY = 91                       # an origin needs 91 days on sale (age_days >= 90)
EVENT_TYPES = ("Sporting", "Cultural", "National", "Religious")
STATIC = ["item_id", "store_id", "dept_id", "cat_id"]
CAL_COLS = ["n_snap", "n_event", *(f"n_{t.lower()}" for t in EVENT_TYPES), "n_weekend", "n_closed"]


def load():
    """wide sales / price (dates x series, NaN before launch), per-day calendar flags, static ids"""
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    dates = pd.date_range(panel.date.min(), panel.date.max())
    sales = panel.pivot(index="date", columns="id", values="sales").reindex(dates).astype("float32")
    price = panel.pivot(index="date", columns="id", values="sell_price").reindex(dates).astype("float32")
    d = panel.groupby("date").agg(closed=("closed", "first"), snap=("snap", "first"), t1=("event_type_1", "first"),
                                  t2=("event_type_2", "first")).reindex(dates)
    cal = pd.DataFrame({"n_snap": d.snap, "n_event": d.t1.notna() | d.t2.notna(),
                        **{f"n_{t.lower()}": (d.t1 == t) | (d.t2 == t) for t in EVENT_TYPES},
                        "n_weekend": dates.dayofweek >= 5, "n_closed": d.closed}, index=dates).astype("int8")
    static = panel.groupby("id")[STATIC].first().reindex(sales.columns)
    return sales, price, cal[CAL_COLS], static


def fwd(x, P):
    """sum over t+1..t+P at row t (NaN if the window is incomplete)"""
    return x.rolling(P, min_periods=P).sum().shift(-P)


def zero_run(open_sales, closed, window, cap):
    """longest run of zero-sale open days inside the window ending at t (closed days neither extend nor break a run), capped"""
    z, c = (open_sales == 0).to_numpy(), closed.to_numpy()
    out = np.full(z.shape, np.nan, "float32")
    for t in range(window - 1, len(z)):
        run = np.zeros(z.shape[1])
        best = np.zeros(z.shape[1])
        for j in range(t - window + 1, t + 1):
            if not c[j]:
                run = np.where(z[j], run + 1, 0)
                np.maximum(best, run, out=best)
        out[t] = np.minimum(best, cap)
    return pd.DataFrame(out, index=open_sales.index, columns=open_sales.columns)


def build(sales, price, cal, static):
    dates, ids = sales.index, sales.columns
    T, N = sales.shape
    closed = cal.n_closed.astype(bool)
    os_ = sales.mask(np.broadcast_to(closed.to_numpy()[:, None], sales.shape))                      # closed days are not demand: masked in rolling stats
    f = {}
    # -- history: uses data <= t only --
    for k in LAGS:
        f[f"lag_{k}"] = sales.shift(k)
    for w in WINDOWS:
        r = os_.rolling(w, min_periods=w - 1)
        f[f"mean_{w}"], f[f"std_{w}"] = r.mean(), r.std()
    z = (os_ == 0).astype("float32").where(os_.notna())
    for w in (28, 91):
        f[f"zero_frac_{w}"] = z.rolling(w, min_periods=w - 1).mean()
    first_date = price.apply(lambda s: s.first_valid_index())
    age = pd.DataFrame((dates.values[:, None] - first_date.values[None, :]).astype("timedelta64[D]").astype(float),
                       index=dates, columns=ids).where(sales.notna())
    last_sale = pd.DataFrame(np.where(sales.to_numpy() > 0, np.arange(T)[:, None], np.nan), index=dates, columns=ids).ffill()
    f["age_days"] = age
    f["days_since_sale"] = (pd.DataFrame(np.arange(T)[:, None] - last_sale.to_numpy(), index=dates, columns=ids)).where(last_sale.notna(), age)
    f["zero_run_91"] = zero_run(os_, closed, ZR_WINDOW, ZR_CAP)
    for P in PS:
        f[f"sum_last_{P}"] = sales.rolling(P, min_periods=P).sum()
        f[f"sum_364_{P}"] = fwd(sales, P).shift(364)                   # window t-363..t-364+P: weekday-aligned, ends <= t
    med = price.rolling(MEDIAN_WINDOW, min_periods=MEDIAN_MIN).median()  # median of prices <= t
    f["price"], f["price_rel_now"] = price, price / med
    # -- future-window covariates: functions of the calendar / price plan, never of sales --
    for P in PS:
        f[f"price_mean_rel_p{P}"] = price.rolling(P, min_periods=P).mean().shift(-P) / med
        f[f"price_min_rel_p{P}"] = price.rolling(P, min_periods=P).min().shift(-P) / med
        f[f"y_p{P}"] = fwd(sales, P)                                   # targets: the only place future sales appear
    date_feats = {"origin_dow": pd.Series(dates.dayofweek, index=dates)}
    for P in PS:
        for c in CAL_COLS:
            date_feats[f"{c}_p{P}"] = fwd(cal[c].astype("float32"), P)
    out = pd.DataFrame({**{k: v.to_numpy("float32").ravel() for k, v in f.items()},
                        **{k: np.repeat(v.to_numpy("float32"), N) for k, v in date_feats.items()}})
    out.insert(0, "date", np.repeat(dates.values, N))
    out.insert(0, "id", np.tile(ids.to_numpy(), T))
    for c in STATIC:
        out[c] = pd.Categorical(np.tile(static[c].to_numpy(), T))
    keep = (out.age_days >= MIN_HISTORY - 1) & out.y_p7.notna()
    return out[keep].reset_index(drop=True)


def columns(P, future_price=True):
    """model inputs for horizon P; future_price=False is the no-future-price ablation (section 4 of the design)"""
    hist = ([f"lag_{k}" for k in LAGS] + [f"{s}_{w}" for w in WINDOWS for s in ("mean", "std")] + ["zero_frac_28", "zero_frac_91",
            "days_since_sale", "zero_run_91", "age_days", f"sum_last_{P}", f"sum_364_{P}", "price", "price_rel_now", "origin_dow"] + STATIC)
    fut = [f"{c}_p{P}" for c in CAL_COLS]
    price_fut = [f"price_mean_rel_p{P}", f"price_min_rel_p{P}"]
    return hist + fut + (price_fut if future_price else [])


if __name__ == "__main__":
    feats = build(*load())
    feats.to_parquet(PROCESSED / "features.parquet")
    print(f"{len(feats)} rows, {feats.id.nunique()} series, origins {feats.date.min().date()}..{feats.date.max().date()}")
    for P in PS:
        cols = columns(P)
        ok = feats.date <= LAST_SALES_DATE - pd.Timedelta(days=P)
        print(f"P={P}: {len(cols)} inputs, {int(ok.sum())} origins with a complete target; NaN share by input (top 5):",
              feats.loc[ok, cols].isna().mean().sort_values(ascending=False).head(5).round(3).to_dict())
    rev = feats[(feats.date >= TEST_START) & (feats.date <= LAST_REVIEW) & (feats.date.dt.dayofweek == 6)]
    print(f"Sunday review origins in the test window: {rev.date.nunique()} dates x {rev.id.nunique()} series = {len(rev)} rows (design: 25 dates)")
