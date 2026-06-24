"""Infrastructure only (design.md, 2026-09-22 pre-registration item 2): the train/validation curve is captured without changing what a model fits to, and
ml/training_curves.py writes it out correctly. No accuracy numbers; synthetic fixtures only."""
import json

import numpy as np
import pandas as pd

from learned import XGB
from training_curves import save


def test_eval_history_has_matching_curves_and_the_selected_round_is_the_vals_best_not_the_trains(feats):
    P = 7
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.7)]
    m = XGB(P, seed=0, max_rounds=200, patience=10).fit(train, train[f"y_p{P}"])
    assert m.eval_history_ is not None
    metric = next(iter(m.eval_history_["train"]))
    tr_curve, va_curve = m.eval_history_["train"][metric], m.eval_history_["val"][metric]
    assert len(tr_curve) == len(va_curve) > 0
    # adding "train" as a second eval set must not change which round is selected: xgboost's early stopping still tracks "val" (the last eval set),
    # so the chosen round is exactly the val curve's best point, not the train curve's
    assert m.n_rounds_ - 1 == int(np.argmin(va_curve))
    assert m.n_rounds_ <= len(va_curve)


def test_a_model_without_a_date_split_has_no_eval_history():
    m = XGB(7, seed=0, columns=["mean_28"], fixed_rounds=10).fit(pd.DataFrame({"mean_28": [1.0, 2.0, 3.0]}), [1.0, 2.0, 3.0])
    assert m.eval_history_ is None


def test_save_writes_the_curve_or_an_unavailable_placeholder(feats, tmp_path):
    P = 7
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.7)]
    m = XGB(P, seed=0, max_rounds=200, patience=10).fit(train, train[f"y_p{P}"])
    out = save(m, tmp_path / "curve.json")
    on_disk = json.loads((tmp_path / "curve.json").read_text())
    assert on_disk == out and out["available"] is True
    assert out["n_rounds"] == m.n_rounds_ and len(out["train"]) == len(out["val"]) > 0

    plain = XGB(7, seed=0, columns=["mean_28"], fixed_rounds=10).fit(pd.DataFrame({"mean_28": [1.0, 2.0, 3.0]}), [1.0, 2.0, 3.0])
    out2 = save(plain, tmp_path / "none.json")
    assert out2 == {"available": False} == json.loads((tmp_path / "none.json").read_text())
