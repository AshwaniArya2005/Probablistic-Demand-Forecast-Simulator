"""Shared synthetic fixtures: small, seeded, no data/raw needed. The world has a normal series, an intermittent one, a fast one
with a 40-day stock-out run, a series that dies (all zero from day 250), and a short series (first sold 200 days before the end)."""
import numpy as np
import pandas as pd
import pytest

from features import CAL_COLS, EVENT_TYPES, STATIC, build

N_DAYS = 760
IDS = ["A_CA_1", "B_CA_1", "C_CA_2", "DEAD_CA_2", "SHORT_CA_3"]
RATES = {"A": 2.0, "B": 0.3, "C": 6.0, "DEAD": 1.0, "SHORT": 1.5}


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-05", periods=N_DAYS)
    weekend = dates.dayofweek >= 5
    sales = pd.DataFrame({i: rng.poisson(RATES[i.split("_")[0]] * np.where(weekend, 1.5, 1.0)) for i in IDS},
                         index=dates).astype("float32")
    sales.loc[dates[300:340], "C_CA_2"] = 0
    sales.loc[dates[250:], "DEAD_CA_2"] = 0
    sales.loc[dates[:N_DAYS - 200], "SHORT_CA_3"] = np.nan
    closed = (dates.month == 12) & (dates.day == 25)
    sales.loc[closed] = np.where(sales.loc[closed].notna(), 0, np.nan)
    week = np.arange(N_DAYS) // 7
    price = pd.DataFrame({i: 3.0 * (1 - 0.2 * (week % 5 == 0)) for i in IDS}, index=dates).astype("float32")
    price = price.where(sales.notna())
    k = np.arange(N_DAYS)
    ev = {t: (k % 120 == 30 * j) for j, t in enumerate(EVENT_TYPES)}
    cal = pd.DataFrame({"n_snap": dates.day <= 10, "n_event": np.any(list(ev.values()), axis=0) | (k % 45 == 0),
                        **{f"n_{t.lower()}": v for t, v in ev.items()}, "n_weekend": weekend, "n_closed": closed},
                       index=dates)[CAL_COLS].astype("int8")
    static = pd.DataFrame({"item_id": ["ITEMX", "ITEMX", "ITEMX", "ITEMD", "ITEMS"], "store_id": [i[-4:] for i in IDS],
                           "dept_id": "D1", "cat_id": "C1"}, index=IDS)[STATIC]
    return sales, price, cal, static


def make_events(dates):
    """named events: two recurring, one that happens once (Gamma)"""
    k = np.arange(len(dates))
    return pd.DataFrame({"Alpha": (k % 50 == 0), "Beta": (k % 120 == 7), "Gamma": (k == 400)}, index=dates).astype("int8")


@pytest.fixture(scope="session")
def world():
    return make_world()


@pytest.fixture(scope="session")
def feats(world):
    return build(*world)
