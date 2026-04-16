"""Model-specific behaviour promised in the Phase 6 pre-registration (design section 12). Hand-computable cases, no accuracy numbers."""
import numpy as np
import pandas as pd
import pytest

from features import columns
from learned import LR, RF, XGB, early_stop_split
from models import REGISTRY

FAMILIES = [LR, RF, XGB]


def test_registered_and_flagged():
    for c in FAMILIES:
        assert REGISTRY[c.name] is c and c.learned and c.kind == "point"
    assert not LR.nonlinear and RF.nonlinear and XGB.nonlinear


def test_normalisation_scale_is_max_of_mean28_and_floor_times_P():
    X = pd.DataFrame({"mean_28": [0.0, 0.01, 2.0]})
    m = LR(7, columns=["mean_28"], normalize=True, floor=1 / 28)
    # floor 1/28 per day: 7/28 = 0.25 for the two below the floor; 2.0 * 7 = 14 above it
    assert m.scale(X) == pytest.approx([0.25, 0.25, 14.0])
    assert LR(14, columns=["mean_28"], floor=0.1).scale(X) == pytest.approx([1.4, 1.4, 28.0])


def test_normalised_model_multiplies_the_scale_back():
    """y = 3 * scale exactly, so the normalised target is the constant 3 and the forecast must be 3 * scale for any row"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"mean_28": rng.uniform(0.2, 5, 500), "x": rng.normal(size=500)})
    m = LR(7, columns=["mean_28", "x"], normalize=True, floor=1 / 28)
    y = 3 * m.scale(X)
    m.fit(X, y)
    Xn = pd.DataFrame({"mean_28": [0.0, 1.0, 4.0], "x": [0.0, 0.0, 0.0]})
    assert m.predict(Xn) == pytest.approx(3 * m.scale(Xn), rel=1e-6)          # includes a row below the floor


def test_linear_regression_clips_at_zero():
    X = pd.DataFrame({"x1": np.linspace(0, 5, 200)})
    m = LR(7, columns=["x1"]).fit(X, 10 - 2 * X.x1)                            # exact line, hits 0 at x1 = 5
    assert m.predict(pd.DataFrame({"x1": [9.0]})).tolist() == [0.0]           # raw prediction is -8
    assert m.predict(pd.DataFrame({"x1": [1.0]})) == pytest.approx([8.0])


@pytest.mark.parametrize("cls", FAMILIES, ids=lambda c: c.name)
def test_unseen_id_level_does_not_break_prediction(cls, feats):
    P = 7
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.6)]
    m = cls(P, seed=0).fit(train, train[f"y_p{P}"])
    new = feats.tail(20).copy()
    for c in ("item_id", "store_id", "dept_id", "cat_id"):
        new[c] = "NEVER_SEEN"
    p = m.predict(new)
    assert p.shape == (20,) and np.isfinite(p).all() and (p >= 0).all()


def test_early_stopping_split_is_temporal_with_an_embargo():
    dates = pd.date_range("2020-01-01", "2020-06-30")                          # daily origins, last fit origin L = 2020-06-30
    tr, va = early_stop_split(dates, P=7)
    edge = pd.Timestamp("2020-06-30") - pd.Timedelta(days=56)                  # 2020-05-05
    assert dates[va].min() == edge + pd.Timedelta(days=1) and dates[va].max() == dates[-1]
    assert dates[tr].max() == edge - pd.Timedelta(days=7)                      # 2020-04-28: its target ends exactly on the edge
    assert (dates[tr] + pd.Timedelta(days=7)).max() < dates[va].min() + pd.Timedelta(days=1), "training and validation target days overlap"
    assert set(tr).isdisjoint(va)
    assert early_stop_split(dates, 7)[1].tolist() == va.tolist(), "split must be deterministic (no random validation set)"


def test_xgb_uses_early_stopping_when_dates_exist_and_records_rounds(feats):
    P = 7
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.7)]
    m = XGB(P, seed=0, max_rounds=200, patience=10).fit(train, train[f"y_p{P}"])
    assert 1 <= m.n_rounds_ <= 200
    plain = XGB(P, seed=0, fixed_rounds=25).fit(train.drop(columns="date"), train[f"y_p{P}"])
    assert plain.n_rounds_ == 25, "without dates a fixed round count is used, never a random split"


def test_rf_training_size_is_bounded_by_the_row_cap_and_depth_cap(feats):
    P = 7
    big = pd.concat([feats] * 8, ignore_index=True)                            # ~ 8x rows so the cap of 20,000 binds
    assert len(big) > 20000
    m = RF(P, seed=0, n_estimators=3, max_depth=5).fit(big, big[f"y_p{P}"])
    assert m.model.max_samples == 20000
    assert max(t.get_depth() for t in m.model.estimators_) <= 5
    small = RF(P, seed=0, n_estimators=3).fit(feats, feats[f"y_p{P}"])
    assert small.model.max_samples == len(feats) < 20000, "the cap never exceeds the fit-set size"


@pytest.mark.parametrize("cls", FAMILIES, ids=lambda c: c.name)
def test_ablation_drop_removes_a_feature_completely(cls, feats):
    """a model fitted without a column gives identical forecasts when that column is scrambled (the column is really gone)"""
    P = 7
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.7)]
    test = feats[feats.date > feats.date.quantile(0.7)]
    for col in (f"sum_364_{P}", "item_id"):
        without = cls(P, seed=0, drop=(col,)).fit(train, train[f"y_p{P}"])
        assert col not in without.columns and len(without.columns) == len(columns(P)) - 1
        scrambled = test.copy()
        scrambled[col] = np.random.default_rng(0).permutation(scrambled[col].to_numpy())
        assert np.array_equal(without.predict(test), without.predict(scrambled))


def test_id_ablation_arms_have_the_pre_registered_columns():
    ids = ["item_id", "store_id", "dept_id", "cat_id"]
    full = set(columns(7))
    assert set(ids) <= full
    assert set(RF(7, drop=("item_id",)).columns) == full - {"item_id"}
    assert set(RF(7, drop=tuple(ids)).columns) == full - set(ids)
