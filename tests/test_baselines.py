"""Baselines against hand-computed values (design section 7)."""
import numpy as np
import pandas as pd
import pytest

from models import MA28, Naive, SeasonalNaive


def X(**cols):
    return pd.DataFrame(cols)


def test_naive_is_the_previous_p_days():
    assert Naive(7).predict(X(sum_last_7=[10, 0, 3])).tolist() == [10, 0, 3]
    assert Naive(14).predict(X(sum_last_14=[21])).tolist() == [21]        # reads the column of its own horizon


def test_ma28_is_28_day_mean_times_p():
    # mean 1.5/day: 10.5 over 7 days, 15 over 10, 21 over 14
    x = X(mean_28=[1.5])
    assert MA28(7).predict(x).tolist() == [10.5]
    assert MA28(10).predict(x).tolist() == [15.0]
    assert MA28(14).predict(x).tolist() == [21.0]


def test_seasonal_naive_is_the_window_364_days_back():
    x = X(sum_364_7=[8, 0, 5], sum_last_7=[10, 9, 9])
    assert SeasonalNaive(7).predict(x).tolist() == [8, 0, 5]              # a genuine zero last year stays zero (no fallback)


def test_seasonal_naive_falls_back_to_naive_only_where_last_year_is_missing():
    x = X(sum_364_7=[8, np.nan, 5], sum_last_7=[10, 9, 9])
    m = SeasonalNaive(7)
    assert m.predict(x).tolist() == [8, 9, 5]
    assert m.fallback_share(x) == pytest.approx(1 / 3)


def test_dead_series_forecast_zero():
    x = X(sum_last_7=[0.0], mean_28=[0.0], sum_364_7=[0.0])
    for m in (Naive(7), MA28(7), SeasonalNaive(7)):
        assert m.predict(x).tolist() == [0.0]


def test_missing_current_history_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        Naive(7).predict(X(sum_last_7=[np.nan]))
    with pytest.raises(ValueError):
        MA28(7).predict(X(mean_28=[np.nan]))
    with pytest.raises(ValueError):
        SeasonalNaive(7).predict(X(sum_364_7=[1.0], sum_last_7=[np.nan]))


def test_baselines_are_stateless_fit_is_a_no_op():
    x, y = X(sum_last_7=[4.0]), [100.0]
    m = Naive(7)
    assert m.fit(x, y) is m and m.predict(x).tolist() == [4.0]
