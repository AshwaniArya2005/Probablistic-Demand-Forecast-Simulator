"""Rolling-origin tuning folds (pre-registered in docs/design.md section 12 before any baseline score was computed).
Each fold = 12 consecutive Sunday review origins. The last fold's P = 14 targets end exactly on the tuning cutoff, so no fold
touches the test window. F2 is the holiday fold: the same calendar season as the test window's holiday peak, a year earlier."""
import pandas as pd

from config import TUNING_CUTOFF

FOLDS = {
    "F1": ("2014-08-31", "2014-11-16"),   # autumn, includes Thanksgiving in its targets
    "F2": ("2014-11-23", "2015-02-08"),   # holiday fold: Thanksgiving, Christmas, New Year, Super Bowl
    "F3": ("2015-03-08", "2015-05-24"),   # spring
    "F4": ("2015-05-31", "2015-08-16"),   # summer (the season of version 1's calibration window)
}
MAX_P = 14


def origins(fold):
    a, b = FOLDS[fold]
    return pd.date_range(a, b, freq="7D")


def assert_tuning_only(dates, P):
    """the guard every tuning-stage script calls: no target window may extend past the tuning cutoff"""
    last = pd.DatetimeIndex(dates).max() + pd.Timedelta(days=P)
    if last > TUNING_CUTOFF:
        raise RuntimeError(f"target window ends {last.date()}, after the tuning cutoff {TUNING_CUTOFF.date()}: test-window data")


def select(feats, fold, P):
    f = feats[feats.date.isin(origins(fold))]
    assert_tuning_only(f.date, P)
    return f


TOUCH_KEY = "Phase 10 primary run"


def touch_logged(key=TOUCH_KEY):
    """True if the test-window touch log (docs/design.md section 14) has a table row for `key`; test-window runs refuse to start without it"""
    from config import ROOT
    text = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    sec = text[text.index("## 14. Test-window touch log"):text.index("## 15.")]
    return any(key in line and line.startswith("|") for line in sec.splitlines())
