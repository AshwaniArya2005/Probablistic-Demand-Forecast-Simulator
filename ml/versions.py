"""Model version schedule and the fit / calibration / use windows of design section 5. Pure date arithmetic, no data."""
import pandas as pd

from config import LAST_REVIEW, TEST_START

EMBARGO_DAYS = 84      # fit targets must end this long before the cutoff; also the calibration window length
K_WEEKS = 8            # retrain cadence
WARMUP_START = TEST_START - pd.Timedelta(weeks=4)

# version -> cutoff c. Version 0 exists only for the four warm-up review dates (design section 5).
CUTOFFS = {0: WARMUP_START, **{i + 1: TEST_START + pd.Timedelta(weeks=K_WEEKS * i) for i in range(4)}}


def review_dates():
    """all Sunday review dates: 4 warm-up dates then the 25 test dates"""
    return pd.date_range(WARMUP_START, LAST_REVIEW, freq="7D")


def fit_end(c):
    """newest target day a version cut off at c may have seen"""
    return c - pd.Timedelta(days=EMBARGO_DAYS)


def fit_mask(origin_dates, c, P):
    """origins whose whole target window (origin+1..origin+P) ends on or before fit_end(c)"""
    return (pd.DatetimeIndex(origin_dates) + pd.Timedelta(days=P)) <= fit_end(c)


def calibration_window(c):
    """origins in [fit_end, c]; it starts exactly where the fit set ends"""
    return fit_end(c), c


def calibration_mask(origin_dates, c, P):
    """calibration origins: at or after fit_end(c) (so their target days are all after every fit target day) and with the
    whole target window observed by c"""
    d = pd.DatetimeIndex(origin_dates)
    return (d >= fit_end(c)) & (d + pd.Timedelta(days=P) <= c)


def use_dates(v):
    """review dates served by version v: [c_v, c_{v+1}), the last version up to the last review date"""
    c = CUTOFFS[v]
    nxt = CUTOFFS.get(v + 1)
    d = review_dates()
    return d[(d >= c) & ((d < nxt) if nxt is not None else (d <= LAST_REVIEW))]


def calibration_sundays(origin_dates, c, P):
    """the Sunday origins of the calibration window: what conformal calibration and sigma estimation use (daily origins overlap and are dependent)"""
    d = pd.DatetimeIndex(origin_dates)
    return calibration_mask(d, c, P) & (d.dayofweek == 6)
