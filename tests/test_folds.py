"""Tuning folds are Sunday-origin, non-overlapping, end by the tuning cutoff, include a Nov-Jan period, and cannot reach the test window."""
import pandas as pd
import pytest

import folds
from config import TEST_START, TUNING_CUTOFF
from folds import FOLDS, MAX_P, assert_tuning_only, origins, select

DAY = pd.Timedelta(days=1)


def test_each_fold_is_twelve_consecutive_sundays():
    for f in FOLDS:
        o = origins(f)
        assert len(o) == 12 and (o.dayofweek == 6).all() and (o.to_series().diff().dropna() == 7 * DAY).all()


def test_folds_are_ordered_and_do_not_overlap():
    o = [origins(f) for f in FOLDS]
    for a, b in zip(o, o[1:]):
        assert a.max() < b.min()


def test_no_fold_target_reaches_past_the_tuning_cutoff():
    for f in FOLDS:
        assert origins(f).max() + MAX_P * DAY <= TUNING_CUTOFF
    assert origins(list(FOLDS)[-1]).max() + MAX_P * DAY == TUNING_CUTOFF, "the last fold ends exactly at the cutoff"
    assert TUNING_CUTOFF < TEST_START


def test_at_least_one_fold_covers_a_nov_to_jan_holiday_period():
    """demand days (first origin + 1 .. last origin + 7) span Thanksgiving week through 3 Jan of some year"""
    covered = [(f, y) for f in FOLDS for y in range(2011, 2016)
               if origins(f).min() + DAY <= pd.Timestamp(f"{y}-11-24") and origins(f).max() + 7 * DAY >= pd.Timestamp(f"{y + 1}-01-03")]
    assert covered, "no fold covers 24 Nov - 3 Jan"


def test_guard_rejects_test_window_dates():
    assert_tuning_only(origins("F4"), 14)                                  # passes
    with pytest.raises(RuntimeError):
        assert_tuning_only([TEST_START], 7)
    with pytest.raises(RuntimeError):
        assert_tuning_only(origins("F4"), 15)                              # one day too long


def test_select_returns_only_fold_origins_and_guards(monkeypatch):
    dates = pd.date_range("2014-08-01", "2014-12-31")
    f = pd.DataFrame({"date": dates, "id": "A"})
    assert set(select(f, "F1", 7).date) == set(origins("F1"))
    monkeypatch.setitem(FOLDS, "BAD", ("2015-11-01", "2016-01-17"))
    with pytest.raises(RuntimeError):
        select(pd.DataFrame({"date": pd.date_range("2015-11-01", "2016-01-31"), "id": "A"}), "BAD", 7)


def test_design_doc_matches_the_folds_and_has_a_touch_log():
    """the pre-registered fold dates in design.md are the ones in code, and the test-window touch log exists"""
    from config import ROOT
    doc = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    for name, (a, b) in FOLDS.items():
        assert f"**{name}** {a} to {b}" in doc, f"design.md does not record {name} as {a} to {b}"
    assert "## 14. Test-window touch log" in doc
