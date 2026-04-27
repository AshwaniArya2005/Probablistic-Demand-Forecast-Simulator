"""Conformal calibration (design.md section 12, Phase 7 pre-registration item 6). Hand-computed offsets, the too-few-scores guard, pooling,
scale round trip, clipping and sorting, exact finite-sample coverage on exchangeable data, and the calibration origins."""
import numpy as np
import pandas as pd
import pytest

from conformal import InsufficientScores, apply_offsets, n_min, offset, scores, segment_offsets, thin
from features import PS
from versions import CUTOFFS, calibration_sundays, fit_end, fit_mask

DAY = pd.Timedelta(days=1)


def test_offset_is_the_kth_smallest_with_k_ceil_n_plus_1_alpha():
    s = list(range(1, 11))                                           # n = 10
    assert offset(s, 0.8) == 9.0                                      # k = ceil(11 * 0.8) = ceil(8.8) = 9
    assert offset(s[::-1], 0.8) == 9.0                                # order does not matter
    assert offset(s, 0.5) == 6.0                                      # k = ceil(5.5) = 6
    assert offset([1.5, -2.0, 0.5, 3.0], 0.5) == 1.5                  # n = 4, k = ceil(5 * 0.5) = 3: sorted -2, 0.5, 1.5, 3.0 -> 1.5


def test_n_min_values():
    assert [n_min(a) for a in (0.80, 0.90, 0.95, 0.99)] == [4, 9, 19, 99]


@pytest.mark.parametrize("alpha", [0.80, 0.90, 0.95, 0.99])
def test_too_few_scores_raises_and_the_minimum_is_allowed(alpha):
    m = n_min(alpha)
    with pytest.raises(InsufficientScores):
        offset(np.arange(m - 1, dtype=float), alpha)
    assert offset(np.arange(m, dtype=float), alpha) == m - 1           # n = n_min: k = n, the maximum, allowed by rule


def test_guard_error_is_a_value_error_and_names_the_requirement():
    with pytest.raises(ValueError, match="need at least 99"):
        offset(np.arange(50.0), 0.99)


def test_thin_sample_warning_thresholds():
    assert thin(24, 0.8) and not thin(25, 0.8)
    assert thin(499, 0.99) and not thin(500, 0.99)
    assert thin(49, 0.9) and not thin(50, 0.9) and thin(99, 0.95) and not thin(100, 0.95)


def test_pooling_uses_the_given_segment_labels():
    sc = [1, 2, 3, 4, 10, 20, 30, 40]
    seg = ["a"] * 4 + ["b"] * 4
    # each segment n = 4, alpha 0.5: k = ceil(5 * 0.5) = 3 -> third smallest of each
    assert segment_offsets(sc, seg, 0.5) == {"a": 3.0, "b": 30.0}
    with pytest.raises(InsufficientScores):
        segment_offsets(sc, seg, 0.99)                                 # 4 scores per segment cannot support 0.99


def test_a_thin_segment_is_reported_not_borrowed_from():
    sc = np.r_[np.arange(200.0), [1.0, 2.0]]
    seg = ["big"] * 200 + ["tiny"] * 2
    with pytest.raises(InsufficientScores):
        segment_offsets(sc, seg, 0.9)                                  # tiny has 2 < 9 scores: error, not a silent pooled value


def test_scale_round_trip_by_hand():
    assert scores([12], [10], [4]) == pytest.approx([0.5])            # (12 - 10) / 4
    out = apply_offsets([[10.0]], ["a"], [4.0], {0.9: {"a": 0.5}}, [0.9])
    assert out.tolist() == [[12.0]]                                    # 10 + 0.5 * 4: the series' own scale multiplies the pooled offset back


def test_offsets_are_clipped_at_zero_and_resorted():
    off = {0.8: {"a": 0.0}, 0.9: {"a": 0.0}, 0.95: {"a": -5.0}}
    out = apply_offsets([[1.0, 2.0, 3.0]], ["a"], [1.0], off, [0.8, 0.9, 0.95])
    assert out.tolist() == [[0.0, 1.0, 2.0]]                           # third becomes -2 -> 0 (clip), then sorted: no crossing


@pytest.mark.parametrize("n,alpha", [(99, 0.9), (19, 0.95), (10, 0.8), (49, 0.5)])
def test_exact_finite_sample_coverage_on_exchangeable_data(n, alpha):
    """a new exchangeable score is at or below the k-th smallest of n with probability exactly k / (n + 1), in [alpha, alpha + 1/(n+1))"""
    rng = np.random.default_rng(0)
    reps = 20000
    hits = 0
    for _ in range(reps):
        cal = rng.normal(size=n)
        hits += rng.normal() <= offset(cal, alpha)
    k = int(np.ceil(round((n + 1) * alpha, 9)))
    exact = k / (n + 1)
    assert alpha <= exact < alpha + 1 / (n + 1) + 1e-12
    assert abs(hits / reps - exact) < 4 * np.sqrt(exact * (1 - exact) / reps)


GRID = pd.date_range("2011-04-29", "2016-05-15")


@pytest.mark.parametrize("P", PS)
@pytest.mark.parametrize("v", list(CUTOFFS))
def test_calibration_origins_are_sundays_from_the_calibration_window_only(v, P):
    c = CUTOFFS[v]
    cal = GRID[calibration_sundays(GRID, c, P)]
    fit = GRID[fit_mask(GRID, c, P)]
    assert len(cal) in (11, 12) and (cal.dayofweek == 6).all(), "weekly Sundays, not daily origins"
    assert cal.min() == fit_end(c) and (cal + P * DAY).max() <= c
    assert cal.min() + DAY > (fit + P * DAY).max(), "no calibration target day may be a fit-set target day"
    assert set(cal).isdisjoint(fit)
