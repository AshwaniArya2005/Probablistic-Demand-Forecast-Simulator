"""Constants shared by every phase; values come from docs/design.md (sections 2, 5)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROCESSED, FIGURES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "docs" / "figures"

STORES = ["CA_1", "CA_2", "CA_3"]
LAST_SALES_DATE = pd.Timestamp("2016-05-22")
TEST_START = pd.Timestamp("2015-11-22")            # first Sunday review date
TUNING_CUTOFF = TEST_START - pd.Timedelta(weeks=12)  # 2015-08-30; velocity/segments/closed-day rule use data <= this
LAST_REVIEW = LAST_SALES_DATE - pd.Timedelta(days=14)  # 2016-05-08
HOLIDAY_END = pd.Timestamp("2016-01-03")           # holiday peak = demand days 2015-11-23..2016-01-03
MIN_HISTORY_DAYS = 730
N_ITEMS, SEED = 100, 42
