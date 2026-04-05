"""Metrics against hand-calculated values. No accuracy numbers are asserted, only definitions."""
import numpy as np
import pandas as pd
import pytest

from metrics import *
from reference_models import poisson_cdf, poisson_quantile


def test_wape_hand():
    # |10-12| + |20-15| = 7 over 30
    assert wape([10, 20], [12, 15]) == pytest.approx(7 / 30)


def test_wape_rejects_zero_actuals():
    with pytest.raises(ValueError):
        wape([0, 0], [1, 1])


def test_mae_rmse_hand():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)                    # (1+0+2)/3
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(np.sqrt(5 / 3))         # sqrt((1+0+4)/3)


def test_mape_drops_zero_actuals():
    # y=0 dropped; |10-12|/10 = 0.2 and |20-10|/20 = 0.5
    assert mape_nonzero([0, 10, 20], [5, 12, 10]) == pytest.approx(0.35)


def test_mase_excludes_zero_and_missing_scale():
    y = [4, 6, 10, 10, 1, 7]
    yhat = [5, 3, 8, 12, 2, 7]
    ids = ["A", "A", "B", "B", "C", "D"]
    scale = pd.Series({"A": 2.0, "B": 4.0, "C": 0.0})          # C has zero scale, D has none
    # A: MAE (1+3)/2 = 2, 2/2 = 1.0. B: MAE (2+2)/2 = 2, 2/4 = 0.5. C and D excluded.
    value, used, excluded = mase(y, yhat, ids, scale)
    assert value == pytest.approx(0.75)
    assert (used, excluded) == (2, 2)


def test_naive_scale_hand():
    d = pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"])
    f = pd.DataFrame({"id": ["A", "A", "A"], "date": d, "y_p7": [5, 9, 3], "sum_last_7": [3, 5, 8]})
    f = pd.concat([f, pd.DataFrame({"id": ["B"], "date": d[:1], "y_p7": [4], "sum_last_7": [4]})])
    s = naive_scale(f, 7, before=pd.Timestamp("2020-01-15"))
    # A: origins whose target ends by 01-15 are 01-01 and 01-08 only; errors |5-3|=2, |9-5|=4 -> 3. B: error 0.
    assert s["A"] == pytest.approx(3.0) and s["B"] == 0.0


def test_pinball_hand():
    # alpha=.9: y=10,q=8 under-forecast by 2 -> 1.8; y=8,q=10 over-forecast by 2 -> 0.2
    assert pinball([10, 8], [8, 10], 0.9) == pytest.approx(1.0)
    assert pinball([10, 8], [8, 10], 0.5) == pytest.approx(0.5 * mae([10, 8], [8, 10]))


@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("lam", [0.7, 3, 9])
def test_pinball_minimised_at_true_quantile(alpha, lam):
    """expected pinball loss under Poisson(lam), computed exactly from the pmf, is minimal at the true alpha-quantile"""
    ks = np.arange(0, 80)
    pmf = np.diff(np.r_[0, [poisson_cdf(lam, k) for k in ks]])

    def expected(q):
        d = ks - q
        return float((pmf * np.maximum(alpha * d, (alpha - 1) * d)).sum())

    q_true = poisson_quantile(lam, alpha)
    best = min(expected(q) for q in range(0, 40))
    assert expected(q_true) == pytest.approx(best, abs=1e-12)
    assert expected(q_true) < expected(q_true + 3)
    assert expected(q_true) <= expected(max(q_true - 3, 0))


def test_scaled_pinball_hand():
    y, q = [10, 8, 3, 5, 9], [8, 10, 3, 1, 9]
    ids = ["A", "A", "B", "B", "C"]
    scale = pd.Series({"A": 2.0, "B": 3.0, "C": 0.0})
    # alpha=.9 rows: 1.8, 0.2, 0, 3.6 (y=5,q=1), 0. A mean 1.0 / 2 = 0.5; B mean 1.8 / 3 = 0.6; C excluded (zero scale)
    value, used, excluded = scaled_pinball(y, q, 0.9, ids, scale)
    assert value == pytest.approx(0.55) and (used, excluded) == (2, 1)


def test_coverages_hand():
    assert coverage_interval([1, 5, 9], [0, 0, 0], [2, 4, 10]) == pytest.approx(2 / 3)     # 5 is above 4
    assert coverage_interval([2], [2], [2]) == 1.0                                          # bounds inclusive
    assert coverage_onesided([1, 2, 3, 4], [2, 2, 2, 5]) == pytest.approx(0.75)


@pytest.mark.parametrize("alpha", [0.8, 0.9, 0.95, 0.99])
def test_onesided_coverage_of_true_quantile_matches_exact_cdf(alpha):
    """with the true quantile as forecast, empirical one-sided coverage equals the exact CDF at that quantile (up to sampling error)
    and is at least alpha: discrete demand gives nominal-or-higher coverage"""
    lam, n = 7, 40000
    y = np.random.default_rng(0).poisson(lam, n)
    q = poisson_quantile(lam, alpha)
    exact = poisson_cdf(lam, q)
    assert exact >= alpha
    se = np.sqrt(exact * (1 - exact) / n)
    assert abs(coverage_onesided(y, np.full(n, q)) - exact) < 4 * se


def test_mase_by_series_hand_and_median():
    y, yhat = [4, 6, 10, 10, 1], [5, 3, 8, 12, 2]
    ids = ["A", "A", "B", "B", "C"]
    r = mase_by_series(y, yhat, ids, pd.Series({"A": 2.0, "B": 4.0, "C": 0.0}))
    assert r["A"] == pytest.approx(1.0) and r["B"] == pytest.approx(0.5) and np.isnan(r["C"])
    assert r.median() == pytest.approx(0.75)                       # median over the two usable series
    # one tiny scale dominates the mean but not the median: MASE 100/0.01 = 10000, 100/100 = 1, 100/100 = 1
    big = mase_by_series([100, 100, 100], [0, 0, 0], ["A", "B", "C"], pd.Series({"A": 0.01, "B": 100.0, "C": 100.0}))
    assert big.mean() == pytest.approx(10002 / 3) and big.median() == pytest.approx(1.0)
