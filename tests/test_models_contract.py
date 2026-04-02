"""Contract every registered model must satisfy (baselines now; Linear Regression, Random Forest, XGBoost, quantile XGBoost join
the registry in later phases and are covered automatically). Runs on the synthetic world: no data/raw needed.
Nothing here asserts an accuracy number."""
import numpy as np
import pandas as pd
import pytest

import models
from features import PS, columns
from models import REGISTRY

PARAMS = [(n, P) for n in REGISTRY for P in PS]
IDS = [f"{n}-P{P}" for n, P in PARAMS]
ALPHAS = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]


@pytest.fixture(scope="module")
def split(feats):
    cut = feats.date.sort_values().iloc[int(len(feats) * 0.7)]
    return cut, feats


def frames(feats, cut, P):
    train = feats[feats.date + pd.Timedelta(days=P) <= cut]        # target windows ended before the cut
    test = feats[feats.date > cut].reset_index(drop=True)
    return train, test


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_shape_finite_nonnegative(name, P, split):
    cut, feats = split
    train, test = frames(feats, cut, P)
    m = REGISTRY[name](P, seed=0).fit(train, train[f"y_p{P}"])
    p = m.predict(test)
    assert p.shape == (len(test),)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert m.predict(test.iloc[0:0]).shape == (0,)


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_fit_does_not_mutate_inputs(name, P, split):
    cut, feats = split
    train, _ = frames(feats, cut, P)
    y = train[f"y_p{P}"].copy()
    before = train.copy()
    REGISTRY[name](P, seed=0).fit(train, y)
    pd.testing.assert_frame_equal(train, before)
    pd.testing.assert_series_equal(train[f"y_p{P}"], y)


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_seeded_determinism(name, P, split):
    cut, feats = split
    train, test = frames(feats, cut, P)
    a = REGISTRY[name](P, seed=7).fit(train, train[f"y_p{P}"]).predict(test)
    b = REGISTRY[name](P, seed=7).fit(train, train[f"y_p{P}"]).predict(test)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_save_load_round_trip(name, P, split, tmp_path):
    cut, feats = split
    train, test = frames(feats, cut, P)
    m = REGISTRY[name](P, seed=3).fit(train, train[f"y_p{P}"])
    m.save(tmp_path / "m.json")
    m2 = models.load(tmp_path / "m.json")
    assert type(m2) is type(m) and m2.P == P
    assert np.array_equal(m.predict(test), m2.predict(test))


def check_causality(m, test, P):
    """a forecast at origin t must not change if (a) everything that is not a declared input (targets, other horizons) is scrambled,
    or (b) other rows, including later origins, are removed or reordered"""
    base = m.predict(test)
    rng = np.random.default_rng(0)
    junk = test.copy()
    for c in junk.columns:
        if c not in columns(P) and c not in ("id", "date") and str(junk[c].dtype) != "category":
            junk[c] = rng.permutation(junk[c].to_numpy()) + rng.integers(1, 999, len(junk))
    assert np.array_equal(m.predict(junk), base), "predictions depend on a column that is not a declared input"
    assert np.array_equal(m.predict(test.iloc[::-1])[::-1], base), "predictions depend on row order"
    for k in (0, len(test) // 2, len(test) - 1):
        assert m.predict(test.iloc[[k]])[0] == base[k], "predictions depend on other rows"


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_causality_prediction_depends_only_on_its_own_inputs(name, P, split):
    cut, feats = split
    train, test = frames(feats, cut, P)
    check_causality(REGISTRY[name](P, seed=0).fit(train, train[f"y_p{P}"]), test, P)


class _ReadsTheTarget(models.Naive):
    """broken on purpose: peeks at the target column"""
    def predict(self, X):
        return X[f"y_p{self.P}"].to_numpy("float64")


class _UsesBatchStatistics(models.Naive):
    """broken on purpose: normalises by the batch, so a row's forecast depends on other rows"""
    def predict(self, X):
        v = super().predict(X)
        return v * (1 + 0.01 * v.mean())


@pytest.mark.parametrize("bad", [_ReadsTheTarget, _UsesBatchStatistics])
def test_causality_check_rejects_broken_models(bad, split):
    cut, feats = split
    _, test = frames(feats, cut, 7)
    with pytest.raises(AssertionError):
        check_causality(bad(7), test, 7)


@pytest.mark.parametrize("name,P", PARAMS, ids=IDS)
def test_dead_and_short_series(name, P, split):
    """a series that died (all-zero history) and one with under a year of history (no seasonal window) still get finite, non-negative forecasts"""
    cut, feats = split
    train, test = frames(feats, cut, P)
    m = REGISTRY[name](P, seed=0).fit(train, train[f"y_p{P}"])
    for sid in ("DEAD_CA_2", "SHORT_CA_3"):
        rows = test[test.id == sid]
        assert len(rows) > 0, f"fixture has no test rows for {sid}"
        p = m.predict(rows)
        assert np.isfinite(p).all() and (p >= 0).all()
    assert test[test.id == "SHORT_CA_3"][f"sum_364_{P}"].isna().all(), "fixture must exercise the missing-seasonal-window path"


QUANTILE = [(n, P) for n, P in PARAMS if REGISTRY[n].kind == "quantile"]


@pytest.mark.parametrize("name,P", QUANTILE, ids=[f"{n}-P{P}" for n, P in QUANTILE])
def test_quantile_models_return_sorted_nonnegative_quantiles(name, P, split):
    cut, feats = split
    train, test = frames(feats, cut, P)
    m = REGISTRY[name](P, seed=0).fit(train, train[f"y_p{P}"])
    q = m.predict_quantiles(test, ALPHAS)
    assert q.shape == (len(test), len(ALPHAS)) and np.isfinite(q).all() and (q >= 0).all()
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
