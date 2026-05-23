"""Phase 9 (docs/design.md section 12, Phase 9 pre-registration): reorder points, safety stock, stockout-risk labels and the constant-CV check,
with hand-computed values, monotonicity, integer ceiling and label boundaries. Nothing here asserts accuracy on project data."""
import numpy as np
import pytest

import policy
from policy import (SERVICE, bin_ratio, ceil_units, cv_check, order_quantity, protection_interval, rop_classic, rop_quantile, safety_stock,
                    sigma_p, stockout_risk)
from quantiles import normal_z


# ---------- reorder points by hand ----------
def test_classic_reorder_point_and_safety_stock_by_hand():
    # yhat 10, sigma_seg 0.5, scale 4, alpha 0.9: 10 + 1.2815515655 * (0.5 * 4) = 12.563 -> ROP 13, SS = 13 - 10 = 3
    rop = rop_classic(10.0, 0.5, 4.0, 0.9)
    assert rop == 13 and safety_stock(rop, 10.0) == 3.0
    assert rop_classic(10.0, 0.5, 4.0, 0.8) == 12                    # 10 + 0.8416 * 2 = 11.683 -> 12
    assert rop_classic(-5.0, 0.5, 4.0, 0.9) == 0                     # never negative


def test_quantile_reorder_point_by_hand():
    assert rop_quantile(12.01) == 13 and rop_quantile(12.0) == 12 and rop_quantile(0.0) == 0 and rop_quantile(0.001) == 1
    assert rop_quantile(12.000000000000002) == 12                    # arithmetic noise must not add a unit
    assert rop_quantile(-0.3) == 0


def test_safety_stock_is_floored_for_display_but_raw_is_kept():
    assert safety_stock(5, 8.0) == 0.0 and safety_stock(5, 8.0, display=False) == -3.0


def test_the_two_methods_agree_when_demand_really_is_normal():
    """protection demand Normal(10, 2^2): the classic reorder point equals the ceiling of the exact normal quantile"""
    for a in SERVICE:
        q = 10.0 + normal_z(a) * 2.0
        assert rop_classic(10.0, 0.5, 4.0, a) == rop_quantile(q)


def test_order_quantity_by_hand():
    assert order_quantity(13, 8) == 5 and order_quantity(13, 20) == 0 and order_quantity([13, 4], [8, 4]).tolist() == [5, 0]


# ---------- integer ceiling and monotonicity ----------
def test_reorder_points_are_integers_and_never_below_the_unrounded_value():
    rng = np.random.default_rng(0)
    yhat, sigma, scale = rng.uniform(0, 30, 500), rng.uniform(0.1, 2, 500), rng.uniform(0.2, 20, 500)
    for a in SERVICE:
        rop = rop_classic(yhat, sigma, scale, a)
        raw = np.maximum(yhat + normal_z(a) * sigma * scale, 0.0)
        assert rop.dtype == np.int64 and (rop >= raw - 1e-9).all() and (rop - raw < 1.0 + 1e-9).all()
    q = rng.uniform(0, 40, 500)
    assert rop_quantile(q).dtype == np.int64 and (rop_quantile(q) >= q - 1e-9).all()
    assert ceil_units([1.0, 1.0000000004, 1.001, 2.5]).tolist() == [1, 1, 2, 3]


def test_reorder_points_are_non_decreasing_in_alpha_and_in_the_inputs():
    rng = np.random.default_rng(1)
    yhat, sigma, scale = rng.uniform(0, 30, 300), rng.uniform(0.1, 2, 300), rng.uniform(0.2, 20, 300)
    by_alpha = np.column_stack([rop_classic(yhat, sigma, scale, a) for a in SERVICE])
    assert (np.diff(by_alpha, axis=1) >= 0).all()
    q = np.sort(rng.uniform(0, 40, (300, 4)), axis=1)                # sorted quantiles at the four service levels
    assert (np.diff(rop_quantile(q), axis=1) >= 0).all()
    for a in SERVICE:
        for base in (dict(y=10.0, s=0.5, c=4.0),):
            f = lambda y, s, c: rop_classic(y, s, c, a)
            assert (np.diff(f(np.linspace(0, 50, 60), base["s"], base["c"])) >= 0).all()      # in yhat
            assert (np.diff(f(base["y"], np.linspace(0.1, 3, 60), base["c"])) >= 0).all()      # in sigma
            assert (np.diff(f(base["y"], base["s"], np.linspace(0.2, 30, 60))) >= 0).all()     # in scale


def test_grids_are_enforced():
    with pytest.raises(ValueError):
        rop_classic(10.0, 0.5, 4.0, 0.75)
    with pytest.raises(ValueError):
        protection_interval(5)
    assert protection_interval(3) == 10 and protection_interval(7) == 14


# ---------- stockout-risk labels ----------
def test_label_boundaries_by_hand():
    # q50 = 4.2 -> Q50 5; q90 = 9.5 -> Q90 10; q99 = 15.1 -> Q99 16
    ip = np.array([4, 5, 9, 10, 16, 17])
    label, over = stockout_risk(ip, 4.2, 9.5, 15.1)
    assert label.tolist() == ["HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW", "LOW"]
    assert over.tolist() == [False, False, False, False, False, True]


def test_an_exact_integer_quantile_is_its_own_cut_point():
    label, _ = stockout_risk([4, 5], 5.0, 9.0, 12.0)
    assert label.tolist() == ["HIGH", "MEDIUM"]                     # Q50 = 5, so IP = 5 is not HIGH


def test_degenerate_quantiles_empty_the_bands_between_them():
    label, over = stockout_risk([0, 1], 0.0, 0.0, 0.0)              # all quantiles 0: never HIGH or MEDIUM; one unit is overstock
    assert label.tolist() == ["LOW", "LOW"] and over.tolist() == [False, True]
    label, _ = stockout_risk([2, 3, 4], 3.0, 3.0, 8.0)              # Q50 = Q90 = 3: MEDIUM is empty
    assert label.tolist() == ["HIGH", "LOW", "LOW"] and "MEDIUM" not in label.tolist()
    label, over = stockout_risk([9, 10], 3.0, 9.0, 9.0)             # Q99 = Q90 = 9: the flag fires above Q90
    assert over.tolist() == [False, True]


def test_risk_never_rises_and_overstock_never_falls_as_stock_rises():
    rng = np.random.default_rng(2)
    rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    for _ in range(200):
        q50, q90, q99 = np.sort(rng.uniform(0, 40, 3))
        label, over = stockout_risk(np.arange(0, 60), q50, q90, q99)
        r = [rank[x] for x in label]
        assert all(a >= b for a, b in zip(r, r[1:])), (q50, q90, q99)
        assert all(a <= b for a, b in zip(over, over[1:]))


def test_unsorted_quantiles_are_rejected():
    with pytest.raises(ValueError):
        stockout_risk([5], 9.0, 4.0, 12.0)
    with pytest.raises(ValueError):
        stockout_risk([5], 4.0, 9.0, 8.0)


# ---------- constant-CV check ----------
def test_bin_ratio_by_hand():
    # scale 1..4, residuals 1, -1, 2, -2: with 2 bins the RMSEs are 1 and 2; with 4 bins 1, 1, 2, 2
    assert bin_ratio([1, -1, 2, -2], [1, 2, 3, 4], bins=2) == pytest.approx(2.0)
    assert bin_ratio([1, -1, 2, -2], [1, 2, 3, 4], bins=4) == pytest.approx(2.0)
    assert bin_ratio([1, 1, 1, 1], [1, 2, 3, 4], bins=2) == pytest.approx(1.0)


def _segment_data(kind, n, seed):
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(1.5, 1.0, n))
    sd = {"constant_cv": 0.3 * scale, "sqrt": 0.5 * np.sqrt(scale), "steep": 0.05 * scale ** 2}[kind]
    return rng.normal(0, sd), scale


def test_constant_cv_check_on_synthetic_segments_of_known_structure():
    e1, s1 = _segment_data("constant_cv", 20000, 0)
    e2, s2 = _segment_data("sqrt", 20000, 1)
    e3, s3 = _segment_data("steep", 20000, 2)
    seg = np.array(["a"] * 20000 + ["b"] * 20000 + ["c"] * 20000)
    res = cv_check(np.r_[e1, e2, e3], np.r_[s1, s2, s3], seg)
    assert res["a"]["form"] == "constant-CV" and res["a"]["ratio_constant_cv"] <= 1.5
    assert res["b"]["ratio_constant_cv"] > 1.5 and res["b"]["form"] == "sqrt-scale" and res["b"]["ratio_sqrt_scale"] <= 1.5
    assert res["c"]["form"] == "constant-CV (both failed)" and res["c"]["ratio_sqrt_scale"] > 1.5
    assert {k: v["n"] for k, v in res.items()} == {"a": 20000, "b": 20000, "c": 20000}


def test_sigma_by_hand_for_each_form():
    e, sc = np.array([2.0, -2.0]), np.array([1.0, 1.0])
    assert sigma_p(e, sc, "constant-CV")(5.0) == pytest.approx(10.0)          # sigma 2 per unit scale, times 5
    sigma_sqrt = sigma_p(np.array([2.0, -2.0]), np.array([4.0, 4.0]), "sqrt-scale")   # error / sqrt(scale) = +-1, so sigma' = 1
    assert sigma_sqrt(9.0) == pytest.approx(3.0)                               # sigma' * sqrt(9)
    assert policy.CV_THRESHOLD == 1.5 and policy.CV_BINS == 4
