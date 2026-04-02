"""Quantile machinery: sorting, and why the target is the quantile of the P-day sum. Exact pmf arithmetic, no simulation noise."""
import numpy as np

from metrics import pinball
from quantiles import sort_quantiles
from reference_models import poisson_quantile


def test_sort_fixes_crossing_hand():
    q = np.array([[5, 3, 4], [1, 2, 2], [9, 8, 7]])
    assert sort_quantiles(q).tolist() == [[3, 4, 5], [1, 2, 2], [7, 8, 9]]


def test_sorted_input_is_unchanged_and_input_not_mutated():
    q = np.array([[1.0, 2.0, 3.0]])
    before = q.copy()
    assert np.array_equal(sort_quantiles(q), q) and np.array_equal(q, before)


def test_rearrangement_never_increases_pinball_loss():
    """monotone rearrangement can only improve the average pinball loss over the quantile set"""
    rng = np.random.default_rng(0)
    alphas = np.array([0.1, 0.5, 0.8, 0.9, 0.95, 0.99])
    y = rng.poisson(5, 500)
    q = rng.normal(5, 3, (500, 6))                               # crossing quantile forecasts
    s = sort_quantiles(q)
    assert (np.diff(s, axis=1) >= 0).all()
    loss = lambda Q: np.mean([pinball(y, Q[:, j], a) for j, a in enumerate(alphas)])
    assert loss(s) <= loss(q) + 1e-12


def test_quantile_of_the_sum_is_not_the_sum_of_daily_quantiles():
    """7 independent Poisson(1) days. Daily P90 is 2 each (CDF(2)=.92 >= .9 > CDF(1)), so summing daily P90s gives 14, but the P90
    of the 7-day total, Poisson(7), is 10 (CDF(9)=.83 < .9 <= CDF(10)=.90). Summing quantiles over-orders."""
    daily = poisson_quantile(1, 0.9)
    total = poisson_quantile(7, 0.9)
    assert (daily, total) == (2, 10)
    assert 7 * daily == 14 and 7 * daily > total

