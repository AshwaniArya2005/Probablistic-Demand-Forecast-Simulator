"""Model version schedule (design section 5): fit-set embargo, calibration window start, use windows, staleness. Pure date arithmetic on a
daily grid of origins, so it needs no data. One real-data test checks the same rules on the actual features table."""
import pandas as pd
import pytest

from config import LAST_REVIEW, TEST_START, TUNING_CUTOFF
from features import PS
from versions import CUTOFFS, EMBARGO_DAYS, calibration_mask, calibration_window, fit_end, fit_mask, review_dates, use_dates

DAY = pd.Timedelta(days=1)
GRID = pd.date_range("2011-04-29", "2016-05-15")            # every possible origin day in the data


def test_schedule_dates():
    d = lambda s: pd.Timestamp(s)
    assert CUTOFFS == {0: d("2015-10-25"), 1: d("2015-11-22"), 2: d("2016-01-17"), 3: d("2016-03-13"), 4: d("2016-05-08")}
    assert all(c.dayofweek == 6 for c in CUTOFFS.values()), "cutoffs are Sunday review dates"
    assert CUTOFFS[1] == TEST_START and CUTOFFS[4] == LAST_REVIEW
    assert fit_end(CUTOFFS[1]) == TUNING_CUTOFF, "version 1 sees exactly the tuning-cutoff data"
    assert fit_end(CUTOFFS[0]) == d("2015-08-02")


@pytest.mark.parametrize("P", PS)
@pytest.mark.parametrize("v", list(CUTOFFS))
def test_fit_set_embargo_holds(v, P):
    c = CUTOFFS[v]
    fit = GRID[fit_mask(GRID, c, P)]
    assert len(fit) > 0
    assert (fit + P * DAY).max() <= c - EMBARGO_DAYS * DAY, "a fit target window extends inside the embargo"
    assert (fit + P * DAY).max() == fit_end(c), "no origin is wasted: the last one ends exactly at the embargo boundary"
    assert fit.max() < c


@pytest.mark.parametrize("P", PS)
@pytest.mark.parametrize("v", list(CUTOFFS))
def test_calibration_window_starts_where_the_fit_set_ends(v, P):
    c = CUTOFFS[v]
    start, end = calibration_window(c)
    assert start == fit_end(c) and end == c and (end - start).days == EMBARGO_DAYS == 84    # 12 weeks
    fit = GRID[fit_mask(GRID, c, P)]
    cal = GRID[calibration_mask(GRID, c, P)]
    assert cal.min() == start
    assert (cal + P * DAY).max() <= c, "calibration targets must be observed by the cutoff"
    assert cal.min() + DAY > (fit + P * DAY).max(), "calibration target days must all come after every fit target day"
    assert set(fit).isdisjoint(cal)


def test_the_embargo_test_can_fail():
    """without the embargo (fit targets allowed up to the cutoff) the same assertion fires"""
    c, P = CUTOFFS[1], 14
    no_embargo = GRID[(GRID + P * DAY) <= c]
    assert (no_embargo + P * DAY).max() > fit_end(c)


def test_use_windows_partition_the_review_dates():
    used = [use_dates(v) for v in CUTOFFS]
    all_used = used[0].append(used[1]).append(used[2]).append(used[3]).append(used[4])
    assert list(all_used) == list(review_dates()), "every review date is served by exactly one version, in order"
    assert len(review_dates()) == 29 and (review_dates() >= TEST_START).sum() == 25
    assert [len(u) for u in used] == [4, 8, 8, 8, 1]
    for v, u in zip(CUTOFFS, used):
        assert u.min() == CUTOFFS[v] and (u >= CUTOFFS[v]).all()


def test_staleness_is_12_to_20_weeks():
    """newest target day a version has seen, measured at each review date it serves"""
    for v in CUTOFFS:
        for r in use_dates(v):
            stale = (r - fit_end(CUTOFFS[v])).days
            assert 84 <= stale < 140, (v, r.date(), stale)


@pytest.mark.realdata
@pytest.mark.parametrize("P", PS)
def test_real_features_respect_the_schedule(P):
    from config import PROCESSED
    F = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", f"y_p{P}"])
    for v, c in CUTOFFS.items():
        fit = F[fit_mask(F.date, c, P)]
        assert len(fit) > 0 and (fit.date + P * DAY).max() <= fit_end(c)
        cal = F[calibration_mask(F.date, c, P)]
        assert cal.date.min() >= fit_end(c) and cal.date.max() + P * DAY <= c


@pytest.mark.parametrize("P", PS)
def test_tuning_fold_versions_mirror_deployment(P):
    """Phase 6 pre-registration: one version per fold with cutoff = the fold's first origin, fit targets ending 84 days earlier,
    a calibration window starting exactly there, and nothing reaching the test window"""
    from folds import FOLDS, origins
    for fold in FOLDS:
        c = origins(fold).min()
        fit = GRID[fit_mask(GRID, c, P)]
        assert (fit + P * DAY).max() == fit_end(c) == c - EMBARGO_DAYS * DAY
        cal = GRID[calibration_mask(GRID, c, P)]
        assert cal.min() == fit_end(c) and (cal + P * DAY).max() <= c
        assert set(fit).isdisjoint(origins(fold)) and fit.max() < origins(fold).min()
        assert (fit + P * DAY).max() < TUNING_CUTOFF and origins(fold).max() + P * DAY <= TUNING_CUTOFF
