"""Phase 10 (docs/design.md sections 9 and 12, Phase 10 definitions; docs/testing.md#simulator): hand-computed scenario, timing, conservation, integer,
limit-case and leakage tests for ml/simulator.py and ml/sim_policies.py. Synthetic arrays only; nothing here asserts a result on project data."""
import numpy as np
import pandas as pd
import pytest

import sim_policies as sp
import simulator as sim
import versions
from simulator import bootstrap_weights, cost_parts, curve_stats, cycle_table, item_sums, matched_inventory, replay

L, R = 3, 7


def _one(demand_by_day, S, reviews=(0, 7, 14), L=L):
    """one series; demand_by_day = {day: units}"""
    T = max(reviews) + L + R + 1
    d = np.zeros((T, 1), "int64")
    for k, v in demand_by_day.items():
        d[k, 0] = v
    return d, np.array(reviews), np.array(S, "int64").reshape(-1, 1)


# ---------- hand-computed scenario ----------
HAND = {1: 1, 2: 2, 4: 3, 5: 1, 7: 2, 8: 1, 9: 4, 11: 2, 12: 5, 13: 1, 14: 3, 15: 2, 16: 1, 18: 3, 19: 2, 21: 4, 22: 3, 23: 1}


def test_hand_computed_path_orders_receipts_and_lost_sales():
    """S = 10 at reviews on days 0, 7, 14; R = 7, L = 3. Written out by hand:
    day 7 closes with 1 on hand -> order 9, arriving before day 11; day 14 closes with 0 -> order 10, arriving before day 18."""
    d, rv, S = _one(HAND, [10, 10, 10])
    rep = replay(d, rv, S, L)
    assert rep.onhand[1:25, 0].tolist() == [9, 7, 7, 4, 3, 3, 1, 0, 0, 0, 7, 2, 1, 0, 0, 0, 0, 7, 5, 5, 1, 0, 0, 0]
    assert {k: int(v) for k, v in enumerate(rep.lost[:, 0]) if v} == {9: 4, 14: 2, 15: 2, 16: 1, 22: 2, 23: 1}
    assert rep.orders[:, 0].tolist() == [0, 9, 10]
    t = cycle_table(rep, d, rv, L, [0, 1, 2])
    assert t["demanded"][:, 0].tolist() == [11, 14, 13] and t["lost"][:, 0].tolist() == [4, 5, 3]
    assert t["onhand"][:, 0].tolist() == [11, 10, 18] and t["stockout"][:, 0].tolist() == [1, 1, 1]
    s = item_sums(t, ["a"], ["a"])
    assert 1 - s["lost"][0] / s["demanded"][0] == pytest.approx(26 / 38)                       # fill rate
    assert s["onhand"][0] / s["days"][0] == pytest.approx(39 / 21)                              # average on-hand per day
    # warm-up excluded: only the cycles of reviews 1 and 2 count
    s2 = item_sums(cycle_table(rep, d, rv, L, [1, 2]), ["a"], ["a"])
    assert (s2["demanded"][0], s2["lost"][0], s2["onhand"][0], s2["cycles"][0], s2["stockout"][0]) == (27, 8, 28, 2, 2)


# ---------- timing ----------
def test_an_order_arrives_before_opening_of_day_t_plus_L_plus_1():
    d, rv, S = _one({10: 1, 11: 1}, [0, 5, 0])            # nothing on hand; review 1 (day 7) orders 5
    rep = replay(d, rv, S, L)
    assert rep.orders[1, 0] == 5
    assert rep.lost[10, 0] == 1 and rep.served[10, 0] == 0          # day t + L = 10: not there yet
    assert rep.served[11, 0] == 1 and rep.onhand[11, 0] == 4        # day t + L + 1 = 11: served from the arrival


def test_the_protection_interval_is_days_t_plus_1_to_t_plus_R_plus_L():
    """S = 9, nothing else in stock. A spike of S + 1 units on day t + R + L = 10 is the last unit the protection interval of review 0 must cover: one unit is lost
    and it is booked to review 0's cycle. The same spike a day later (t + R + L + 1 = 11) belongs to review 1's cycle instead."""
    for day, cycle in ((10, 0), (11, 1)):
        d, rv, S = _one({day: 10}, [9, 9, 9])
        rep = replay(d, rv, S, L)
        assert cycle_table(rep, d, rv, L, [0, 1, 2])["lost"][:, 0].tolist() == [1 if c == cycle else 0 for c in range(3)], day
        assert rep.lost[:, 0].nonzero()[0].tolist() == [day]
    d, rv, S = _one({10: 9}, [9, 9, 9])
    assert replay(d, rv, S, L).lost.sum() == 0                                        # a spike of exactly S is served


def test_one_unit_a_day_is_covered_by_S_equal_to_P_and_not_by_one_less():
    rv = np.arange(0, 7 * 6, 7)
    d = np.ones((rv[-1] + L + R + 1, 1), "int64")
    assert replay(d, rv, np.full((len(rv), 1), 10), L).lost.sum() == 0             # S = P = 10: never short
    rep = replay(d, rv, np.full((len(rv), 1), 9), L)
    assert cycle_table(rep, d, rv, L, range(len(rv)))["lost"][0, 0] == 1 and rep.lost[rv[0] + L + R, 0] == 1


def test_longer_lead_time_keeps_the_pipeline_in_the_inventory_position():
    """L = 7 = R: the order of the previous review is still on order at the next review; the position counts it, so no double ordering"""
    rv = np.array([0, 7, 14])
    d = np.zeros((rv[-1] + 7 + R + 1, 1), "int64")
    d[1:8, 0] = 2                                             # 14 units in week 1
    rep = replay(d, rv, np.full((3, 1), 20), 7)
    assert rep.orders[1, 0] == 14                              # 20 - (6 on hand + 0 on order); arrives before day 15
    assert rep.orders[2, 0] == 0                               # day 14: on hand 6 + 14 on order = 20 = S, so order nothing


# ---------- conservation and integers ----------
def _random_case(seed, K=8, N=6):
    rng = np.random.default_rng(seed)
    rv = np.arange(0, 7 * K, 7)
    d = rng.poisson(rng.uniform(0.2, 6, N), size=(rv[-1] + L + R + 1, N)).astype("int64")
    S = rng.integers(0, 60, size=(K, N))
    return d, rv, S


@pytest.mark.parametrize("seed", range(6))
def test_conservation_invariants(seed):
    d, rv, S = _random_case(seed)
    rep = replay(d, rv, S, L)
    d0, end = int(rv[0]) + 1, int(rv[-1]) + L + R
    dem = d[d0:end + 1]
    assert (dem == rep.served[d0:end + 1] + rep.lost[d0:end + 1]).all()                   # demand = sales + lost sales
    assert (rep.onhand >= 0).all() and (rep.served >= 0).all() and (rep.lost >= 0).all() # inventory never negative
    prev = np.vstack([S[0][None, :], rep.onhand[d0:end]])
    receipts = rep.onhand[d0:end + 1] + rep.served[d0:end + 1] - prev                     # start + receipts - sales = end
    assert (receipts >= 0).all()
    expected = np.zeros_like(receipts)
    for k in range(1, len(rv)):
        a = int(rv[k]) + L + 1 - d0
        if a <= end - d0:
            expected[a] += rep.orders[k]
    assert (receipts == expected).all()                                                    # nothing received that was not ordered L days earlier
    assert (rep.orders >= 0).all() and (rep.orders[0] == 0).all()


def test_everything_is_integer_and_non_integer_levels_are_rejected():
    d, rv, S = _random_case(0)
    rep = replay(d, rv, S, L)
    for a in (rep.onhand, rep.lost, rep.served, rep.orders):
        assert a.dtype == np.int64
    with pytest.raises(TypeError):
        replay(d, rv, S.astype(float), L)
    with pytest.raises(TypeError):
        replay(d.astype(float), rv, S, L)
    with pytest.raises(ValueError):
        replay(d, rv, -S - 1, L)
    with pytest.raises(ValueError):
        replay(d, np.array([0, 6, 13, 20, 27, 34, 41, 48]), S, L)                           # reviews must be exactly R apart
    with pytest.raises(ValueError):
        replay(d[:20], rv, S, L)                                                            # demand must cover the last cycle


# ---------- limit cases ----------
def test_limit_cases():
    d, rv, S = _random_case(1)
    K, N = S.shape
    zero = replay(d, rv, np.zeros((K, N), "int64"), L)
    assert zero.served.sum() == 0 and zero.orders.sum() == 0 and zero.onhand.sum() == 0    # S = 0: fill rate 0
    t = cycle_table(zero, d, rv, L, range(K))
    assert t["lost"].sum() == t["demanded"].sum() > 0
    big = replay(d, rv, np.full((K, N), 10 ** 6), L)
    tb = cycle_table(big, d, rv, L, range(K))
    assert tb["lost"].sum() == 0 and tb["stockout"].sum() == 0                             # very large S: fill rate 1, no stockouts
    flat = np.tile(S[0], (K, 1))                                 # constant S: with no demand nothing is ever ordered
    none = replay(np.zeros_like(d), rv, flat, L)
    tn = cycle_table(none, np.zeros_like(d), rv, L, range(K))
    assert none.served.sum() == 0 and tn["stockout"].sum() == 0 and tn["lost"].sum() == 0  # zero demand: no sales, no stockouts
    assert none.orders.sum() == 0 and (tn["onhand"] > 0).any()                              # ... and stock just sits (holding accrues)


def test_closed_days_replay_zero_demand_while_holding_cost_accrues():
    d, rv, S = _one({}, [10, 10, 10])                       # every day, closed days included, has zero demand
    rep = replay(d, rv, S, L)
    t = cycle_table(rep, d, rv, L, [0, 1, 2])
    hold, stock = cost_parts([10.0], t["onhand"][:, 0].sum(), t["lost"][:, 0].sum(), 9)
    assert t["lost"].sum() == 0 and t["onhand"].sum() == 10 * 21 and stock == 0 and hold == pytest.approx(0.7 * 10 * 0.25 / 365 * 210)


def test_cost_by_hand():
    h = 0.7 * 10.0 * 0.25 / 365.0                          # holding per unit-day at price 10
    hold, stock = cost_parts([10.0, 20.0], np.array([100, 50]), np.array([2, 1]), 9)
    assert hold == pytest.approx([100 * h, 50 * 2 * h])
    assert stock == pytest.approx([9 * h * 7 * 2, 9 * 2 * h * 7 * 1])                       # rho x holding per unit over 7 days, per lost unit


def test_naive_rule_with_c_one_is_ceil_mu28_times_P():
    tab = pd.DataFrame({"id": ["a", "b", "c"], "date": pd.Timestamp("2015-11-22"), "mu28": [1.2, 1.21, 0.0]})
    assert sp.s_naive(tab, 1.0, 10).tolist() == [12, 13, 0]                                  # 12.0 stays 12 (noise-safe), 12.1 -> 13
    assert sp.s_naive(tab, 1.5, 10).tolist() == [18, 19, 0]                                  # 18.0 -> 18, 18.15 -> 19


# ---------- leakage ----------
def test_decisions_at_or_before_review_t_do_not_depend_on_later_demand():
    d, rv, S = _random_case(2)
    base = replay(d, rv, S, L)
    for j in range(len(rv)):
        d2 = d.copy()
        d2[int(rv[j]) + 1:] += 5                              # perturb every day after review j
        again = replay(d2, rv, S, L)
        assert (again.orders[:j + 1] == base.orders[:j + 1]).all(), j
        assert (again.onhand[:int(rv[j]) + 1] == base.onhand[:int(rv[j]) + 1]).all()


def test_policy_levels_are_built_from_forecast_tables_not_from_simulated_stock():
    """the level builders take a forecast table and a calibration table; changing anything a simulation could produce cannot change them"""
    import inspect
    for fn in (sp.s_naive, sp.s_point, sp.s_quantile, sp.s_posthoc, sp.all_levels):
        assert not {"onhand", "lost", "served", "rep", "replay", "demand", "sales"} & set(inspect.signature(fn).parameters)
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-11-01", periods=3, freq="7D")
    ids = [f"i{j}_CA_1_evaluation" for j in range(6)]
    seg = dict(zip(ids, ["low", "low", "mid", "mid", "high", "high"]))
    tab = pd.DataFrame([(i, d) for d in dates for i in ids], columns=["id", "date"])
    tab["segment"] = tab.id.map(seg)
    tab["mu28"] = rng.uniform(0.5, 5, len(tab))
    tab["scale"] = tab.mu28 * 10
    tab["yhat"] = tab.scale * rng.uniform(0.8, 1.2, len(tab))
    q = np.sort(tab.scale.to_numpy()[:, None] * rng.uniform(0.5, 2.0, (len(tab), 4)), axis=1)
    for j, c in enumerate(("q80", "q90", "q95", "q99")):
        tab[c] = q[:, j]
    cal = pd.DataFrame({"id": np.tile(ids, 300), "segment": np.tile([seg[i] for i in ids], 300)})
    cal["scale"] = rng.uniform(1, 40, len(cal))
    cal["yhat"] = cal.scale * rng.uniform(0.8, 1.2, len(cal))
    cal["y"] = rng.poisson(cal.yhat).astype(float)
    S1, forms = sp.all_levels(tab, cal, 10, (1.0, 1.5))
    S2, _ = sp.all_levels(tab, cal, 10, (1.0, 1.5))
    assert set(S1) == {("naive", 1.0), ("naive", 1.5)} | {(p, a) for p in ("point", "quantile", "posthoc") for a in (0.8, 0.9, 0.95, 0.99)}
    assert all(v.shape == (3, 6) and v.dtype == np.int64 and (v >= 0).all() and (v == S2[k]).all() for k, v in S1.items())      # integers, deterministic
    for p in ("point", "quantile", "posthoc"):
        by_alpha = np.stack([S1[(p, a)] for a in (0.8, 0.9, 0.95, 0.99)])
        assert (np.diff(by_alpha, axis=0) >= 0).all(), p                                     # S never falls as the service level rises
    assert set(forms) == {"low", "mid", "high"}


def test_the_version_schedule_selects_the_model_for_each_review_date():
    dates = versions.review_dates()
    v = sp.version_of(dates)
    assert len(dates) == 29 and v[:4] == [0] * 4 and sorted(set(v)) == [0, 1, 2, 3, 4] and v == sorted(v)
    for d, x in zip(dates, v):
        c = versions.CUTOFFS[x]
        nxt = versions.CUTOFFS.get(x + 1)
        assert d >= c and (nxt is None or d < nxt)


# ---------- matched service and the bootstrap ----------
def test_matched_inventory_by_hand():
    fill, inv = [0.90, 0.95, 0.99], [10.0, 20.0, 40.0]
    assert matched_inventory(0.925, fill, inv) == pytest.approx(15.0)
    assert matched_inventory(0.97, fill, inv) == pytest.approx(30.0)
    assert matched_inventory(0.90, fill, inv) == pytest.approx(10.0) and matched_inventory(0.99, fill, inv) == pytest.approx(40.0)
    assert np.isnan(matched_inventory(0.85, fill, inv)) and np.isnan(matched_inventory(0.995, fill, inv))       # no extrapolation
    assert matched_inventory(0.925, [0.99, 0.90, 0.95], [40.0, 10.0, 20.0]) == pytest.approx(15.0)                # order of the curve does not matter
    got = matched_inventory(np.array([0.925, 0.85]), np.array([fill, fill]), np.array([inv, inv]))                 # vectorised over resamples
    assert got[0] == pytest.approx(15.0) and np.isnan(got[1])


def test_bootstrap_weights_are_item_clusters_and_reproducible():
    W = bootstrap_weights(100, B=500, seed=0)
    assert W.shape == (500, 100) and (W.sum(1) == 100).all() and (W == bootstrap_weights(100, B=500, seed=0)).all()
    ids = [f"item{i}_CA_{s}" for i in range(4) for s in (1, 2, 3)]
    items = [f"item{i}" for i in range(4)]
    table = {k: np.arange(12, dtype=float).reshape(1, 12) + 1 for k in ("demanded", "lost", "onhand", "stockout")}
    s = item_sums(table, [i.rsplit("_CA_", 1)[0] for i in ids], items)
    assert s["cycles"].tolist() == [3, 3, 3, 3] and s["days"].tolist() == [21] * 4
    assert s["demanded"].tolist() == [6, 15, 24, 33]                                        # an item's three store-series are summed together
    w = np.array([[1.0, 0, 2, 1]])                                                          # item 1 dropped, item 2 counted twice
    st = curve_stats(w, dict(s, lost=np.array([1.0, 1, 1, 1])))
    assert st["fill"][0] == pytest.approx(1 - 4 / (6 + 48 + 33))


# ---------- an independent reference implementation ----------
def _reference(demand, reviews, S, L, R=7):
    """one series at a time, event-style, written differently from simulator.replay: a dict of arrivals by day and a running inventory position"""
    out_on, out_lost, out_orders = {}, {}, {}
    on, pipeline = int(S[0]), {}
    for d in range(reviews[0] + 1, reviews[-1] + L + R + 1):
        on += pipeline.pop(d, 0)
        served = min(on, int(demand[d]))
        out_lost[d], on = int(demand[d]) - served, on - served
        out_on[d] = on
        if d in reviews[1:]:
            k = list(reviews).index(d)
            position = on + sum(pipeline.values())
            q = max(int(S[k]) - position, 0)
            out_orders[k] = q
            pipeline[d + L + 1] = pipeline.get(d + L + 1, 0) + q
    return out_on, out_lost, out_orders


@pytest.mark.parametrize("lead", [0, 3, 7])
def test_replay_matches_an_independent_implementation_series_by_series(lead):
    rng = np.random.default_rng(10 + lead)
    K, N = 9, 5
    rv = np.arange(3, 3 + 7 * K, 7)
    d = rng.poisson(rng.uniform(0.1, 8, N), size=(rv[-1] + lead + R + 1, N)).astype("int64")
    S = rng.integers(0, 70, size=(K, N))
    rep = replay(d, rv, S, lead)
    for j in range(N):
        on, lost, orders = _reference(d[:, j], list(rv), S[:, j], lead)
        assert [int(rep.onhand[t, j]) for t in on] == list(on.values())
        assert [int(rep.lost[t, j]) for t in lost] == list(lost.values())
        assert {k: int(rep.orders[k, j]) for k in orders} == orders


def test_matched_reduction_in_both_anchors_by_hand():
    fq, iq = [0.90, 0.95, 0.99], [10.0, 20.0, 40.0]                  # headline policy's curve
    fc, ic = [0.92, 0.97, 0.995], [20.0, 40.0, 80.0]                  # comparator's curve
    # q-anchored: comparator inventory at q's fills 0.90 (outside: NaN), 0.95 (between 0.92 and 0.97: 20 + 0.6 * 20 = 32), 0.99 (40 + 0.8 * 40 = 72)
    qa = sim.matched_reduction(fq, iq, fc, ic, "q")
    assert np.isnan(qa[0]) and qa[1] == pytest.approx(1 - 20 / 32) and qa[2] == pytest.approx(1 - 40 / 72)
    # c-anchored: q's inventory at the comparator's fills 0.92 (10 + 0.4 * 10 = 14), 0.97 (20 + 0.5 * 20 = 30), 0.995 (outside: NaN)
    ca = sim.matched_reduction(fq, iq, fc, ic, "c")
    assert ca[0] == pytest.approx(1 - 14 / 20) and ca[1] == pytest.approx(1 - 30 / 40) and np.isnan(ca[2])
    with pytest.raises(ValueError):
        sim.matched_reduction(fq, iq, fc, ic, "x")


# ---------- the driver: units, sub-periods, the touch-log guard ----------
def _tab_cal(dates, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"i{j}_CA_1_evaluation" for j in range(6)]
    seg = dict(zip(ids, ["low", "low", "mid", "mid", "high", "high"]))
    tab = pd.DataFrame([(i, d) for d in dates for i in ids], columns=["id", "date"])
    tab["segment"] = tab.id.map(seg)
    tab["mu28"] = rng.uniform(0.5, 5, len(tab))
    tab["scale"] = tab.mu28 * 10
    tab["yhat"] = tab.scale * rng.uniform(0.8, 1.2, len(tab))
    q = np.sort(tab.scale.to_numpy()[:, None] * rng.uniform(0.5, 2.0, (len(tab), 4)), axis=1)
    for j, c in enumerate(("q80", "q90", "q95", "q99")):
        tab[c] = q[:, j]
    cal = pd.DataFrame({"id": np.tile(ids, 300), "segment": np.tile([seg[i] for i in ids], 300)})
    cal["scale"] = rng.uniform(1, 40, len(cal))
    cal["yhat"] = cal.scale * rng.uniform(0.8, 1.2, len(cal))
    cal["y"] = rng.poisson(cal.yhat).astype(float)
    return tab, cal, ids


def test_levels_from_two_model_units_equal_one_unit_when_they_share_a_calibration():
    import sim_run as sr
    dates = pd.date_range("2015-11-01", periods=6, freq="7D")
    tab, cal, ids = _tab_cal(dates)
    one = sr.Run("one", tab, [cal], [list(dates)], dates, 2)
    two = sr.Run("two", tab, [cal, cal], [list(dates[:3]), list(dates[3:])], dates, 2)
    a, fa = sr.levels_for(one, (1.0, 2.0), ids)
    b, fb = sr.levels_for(two, (1.0, 2.0), ids)
    assert set(a) == set(b) and all((a[k] == b[k]).all() and b[k].shape == (6, 6) for k in a) and set(fb) == {0, 1}
    swapped = sr.Run("bad", tab, [cal, cal], [list(dates[3:]), list(dates[:3])], dates, 2)
    with pytest.raises(AssertionError):
        sr.levels_for(swapped, (1.0,), ids)                  # units must cover the review dates in order, once each


def test_each_unit_uses_its_own_calibration():
    import sim_run as sr
    dates = pd.date_range("2015-11-01", periods=4, freq="7D")
    tab, cal, ids = _tab_cal(dates)
    wide = cal.assign(y=cal.yhat + 30.0)                     # a calibration window with much larger errors
    run = sr.Run("mixed", tab, [cal, wide], [list(dates[:2]), list(dates[2:])], dates, 1)
    lv, _ = sr.levels_for(run, (1.0,), ids)
    plain, _ = sr.levels_for(sr.Run("plain", tab, [cal], [list(dates)], dates, 1), (1.0,), ids)
    assert (lv[("point", 0.99)][:2] == plain[("point", 0.99)][:2]).all()               # the first unit is unchanged
    assert (lv[("point", 0.99)][2:] >= plain[("point", 0.99)][2:]).all() and lv[("point", 0.99)][2:].sum() > plain[("point", 0.99)][2:].sum()   # the second unit's wider errors widen sigma
    assert (lv[("quantile", 0.9)] == plain[("quantile", 0.9)]).all()                   # quantile levels do not read the calibration window


def test_peak_sub_period_is_assigned_by_cycle_start_day():
    import sim_run as sr
    dates = pd.date_range("2015-12-13", periods=5, freq="7D")            # reviews 13, 20, 27 Dec, 3, 10 Jan; cycles start review + 4 days
    tab, _, _ = _tab_cal(dates)
    run = sr.Run("x", tab, [], [], dates, 0, pd.Timestamp("2016-01-03"))
    assert sr.peak_mask(run).tolist() == [True, True, True, False, False]      # 27 Dec + 4 = 31 Dec is peak; 3 Jan + 4 = 7 Jan is rest
    assert sr.peak_mask(sr.Run("y", tab, [], [], dates, 0, None)).all()


def test_the_touch_log_guard_needs_a_table_row_in_section_14(tmp_path, monkeypatch):
    import config
    import folds
    (tmp_path / "docs").mkdir()
    text = "## 14. Test-window touch log\n\n| Date | Phase | What |\n|---|---|---|\n| (no entries) | | |\n\nPhase 10 primary run is mentioned in prose only.\n\n## 15. Testing\n"
    (tmp_path / "docs" / "design.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert folds.touch_logged() is False                                   # prose or a header does not count
    (tmp_path / "docs" / "design.md").write_text(text.replace("| (no entries) | | |", "| 2026-09-22 | 10 | Phase 10 primary run: forecast tables and replay | no |"), encoding="utf-8")
    assert folds.touch_logged() is True
