"""Leakage and correctness of ml/features.py on the synthetic world (fast) and on the real panel (marked realdata).
1. brute-force reference: every feature at sampled (series, origin) pairs recomputed with plain loops
2. perturbation: scramble sales / prices after an origin t0 and rebuild; nothing observable at t <= t0 may change, except targets and
   future-window price columns whose windows reach past t0 (and those must change, or the test is blind)."""
import numpy as np
import pandas as pd
import pytest

from config import *
from features import *


@pytest.fixture(scope="module", params=["synthetic", pytest.param("real", marks=pytest.mark.realdata)])
def data(request, world):
    if request.param == "synthetic":
        sales, price, cal, static = world
        t0 = sales.index[450]
    else:
        sales, price, cal, static = load()
        t0 = pd.Timestamp("2014-06-15")
    F = build(sales, price, cal, static)
    return sales, price, cal, static, F, t0


def test_structure(data):
    sales, price, cal, static, F, t0 = data
    allowed = {c for P in PS for c in columns(P)}
    assert not any(c.startswith("y_") for c in allowed), "target in model inputs"
    assert allowed <= set(F.columns)
    for P in PS:
        pf = {f"price_mean_rel_p{P}", f"price_min_rel_p{P}"}
        assert set(columns(P)) - set(columns(P, future_price=False)) == pf
    assert "segment" not in F.columns, "segment labels use data after early origins: reporting/pooling only"
    assert set(cal.index[cal.n_closed == 1].strftime("%m-%d")) == {"12-25"}, "closed flag must be the Dec 25 calendar rule"
    assert (F.age_days >= MIN_HISTORY - 1).all()


def ref(sales, price, cal, sid, t, P):
    s, p = sales[sid], price[sid]
    d = lambda k: t + pd.Timedelta(days=k)
    op = lambda x: not cal.n_closed[x] and not np.isnan(s[x])
    r = {f"lag_{k}": s[d(-k)] for k in LAGS}
    for w in WINDOWS:
        v = np.array([s[d(-k)] for k in range(w) if op(d(-k))])
        r[f"mean_{w}"], r[f"std_{w}"] = v.mean(), v.std(ddof=1)
    for w in (28, 91):
        v = np.array([s[d(-k)] for k in range(w) if op(d(-k))])
        r[f"zero_frac_{w}"] = (v == 0).mean()
    r["age_days"] = (t - p.first_valid_index()).days
    sold = s[(s > 0) & (s.index <= t)]
    r["days_since_sale"] = (t - sold.index[-1]).days if len(sold) else r["age_days"]
    run = best = 0
    for k in range(ZR_WINDOW - 1, -1, -1):                       # oldest to newest
        if cal.n_closed[d(-k)]:
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
    for c in CAL_COLS:
        r[f"{c}_p{P}"] = sum(cal[c][x] for x in fut)
    r[f"price_mean_rel_p{P}"] = np.mean([p[x] for x in fut]) / med
    r[f"price_min_rel_p{P}"] = np.min([p[x] for x in fut]) / med
    r[f"y_p{P}"] = sum(s[x] for x in fut)
    return r


def test_brute_force_reference(data):
    sales, price, cal, static, F, t0 = data
    ok = F[F.date <= sales.index[-1] - pd.Timedelta(days=14)]
    md = ok.date.dt.strftime("%m-%d")
    picks = list(ok.sample(40, random_state=1).index)
    picks += list(ok[md.between("12-20", "12-30")].sample(8, random_state=2).index)     # windows with a closed day
    for sid in ("DEAD_CA_2", "SHORT_CA_3", "C_CA_2", "FOODS_2_101_CA_2_evaluation"):
        s = ok[ok.id == sid]
        picks += list(s.sample(min(5, len(s)), random_state=3).index)
    for ix in picks:
        row = F.loc[ix]
        for P in PS:
            for k, v in ref(sales, price, cal, row.id, row.date, P).items():
                assert np.isclose(row[k], v, rtol=1e-4, atol=1e-4, equal_nan=True), (row.id, row.date.date(), P, k, row[k], v)


def _compare(F, F2, cols, mask):
    a, b = F[mask][["id", "date", *cols]], F2[mask][["id", "date", *cols]]
    assert a[["id", "date"]].equals(b[["id", "date"]]), "row set changed"
    num = [c for c in cols if str(F[c].dtype) != "category"]
    return a[num].fillna(-9e9).to_numpy(), b[num].fillna(-9e9).to_numpy()


def _scramble(frame, kind, t0, rng):
    m = frame.index > t0
    x = frame.copy()
    vals = rng.integers(0, 10, x[m].shape) if kind == "sales" else frame[m].to_numpy() * rng.uniform(0.5, 1.5, x[m].shape)
    x.loc[m] = np.where(frame[m].notna(), vals.astype("float32"), np.nan)
    return x


def test_perturbation_leaks(data):
    sales, price, cal, static, F, t0 = data
    rng = np.random.default_rng(0)
    hist = [c for c in F.columns if c not in ("id", "date") and not c.startswith("y_")]
    ycols = [c for c in F.columns if c.startswith("y_")]
    pfut = [c for c in F.columns if c.startswith(("price_mean_rel_p", "price_min_rel_p"))]
    past, before = F.date <= t0, F.date <= t0 - pd.Timedelta(days=14)
    near = past & ~before
    F2 = build(_scramble(sales, "sales", t0, rng), price, cal, static)                   # sales after t0 scrambled
    a, b = _compare(F, F2, hist, past)
    assert np.array_equal(a, b), "a feature at t <= t0 changed when sales after t0 were scrambled"
    a, b = _compare(F, F2, ycols, before)
    assert np.array_equal(a, b), "targets with windows entirely <= t0 changed"
    a, b = _compare(F, F2, ycols, near)
    assert not np.array_equal(a, b), "blind test: targets reaching past t0 should have changed"
    a, b = _compare(F, F2, hist, F.date > t0 + pd.Timedelta(days=30))
    assert not np.array_equal(a, b), "blind test: history after t0 should have changed"
    F3 = build(sales, _scramble(price, "price", t0, rng), cal, static)                   # prices after t0 scrambled
    a, b = _compare(F, F3, [c for c in hist if c not in pfut], past)
    assert np.array_equal(a, b), "a non-future-price feature at t <= t0 changed when prices after t0 were scrambled"
    a, b = _compare(F, F3, pfut, before)
    assert np.array_equal(a, b), "future-price windows entirely <= t0 changed"
    a, b = _compare(F, F3, pfut, near)
    assert not np.array_equal(a, b), "blind test: future-price windows reaching past t0 should have changed"
    a, b = _compare(F, F3, ycols, past)
    assert np.array_equal(a, b), "prices must not change targets"


@pytest.mark.realdata
def test_load_matches_raw_calendar():
    """load() calendar flags against calendar.csv for every day, re-derived independently"""
    raw = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"]).set_index("date")
    cal = load()[2]
    raw = raw.loc[cal.index]
    assert (cal.n_snap.to_numpy() == raw.snap_CA.to_numpy()).all()
    assert (cal.n_weekend.to_numpy() == raw.weekday.isin(["Saturday", "Sunday"]).astype(int).to_numpy()).all()
    for t in EVENT_TYPES:
        exp = ((raw.event_type_1 == t) | (raw.event_type_2 == t)).astype(int)
        assert (cal[f"n_{t.lower()}"].to_numpy() == exp.to_numpy()).all()
    assert (cal.n_event.to_numpy() == (raw.event_name_1.notna() | raw.event_name_2.notna()).astype(int).to_numpy()).all()
