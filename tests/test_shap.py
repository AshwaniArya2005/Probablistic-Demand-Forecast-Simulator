"""Phase 8 explanations (design.md section 12). Exact per-quantile contributions from sliced single-target boosters (XGBoost 3.4.1's joint
`pred_contribs`, and the `shap` package that wraps it, do not reconcile with the joint `predict`, which returns sorted tree sums), unit conversion,
themes, sign-to-word agreement, wording rules and the guards. Nothing here asserts accuracy."""
import re

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import explain
from config import TEST_START
from explain import (BANNED, THEMES, VERB_DOWN, VERB_UP, clauses, explain_rows, price_words, ratio_to_units, sentence, theme_contributions, theme_of)
from features import PS, columns
from learned import XGBQ

ALPHAS = [0.10, 0.50, 0.80, 0.90, 0.95, 0.99]


@pytest.fixture(scope="module")
def crossing_model():
    """noisy data and few rounds: the six per-target tree sums cross on about 1% of rows, so the rank-source logic is exercised"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 5)).astype("float32"), columns=[f"f{i}" for i in range(5)])
    y = np.maximum(0, 3 + 2 * X.f0 + X.f1 * X.f2 + rng.poisson(1, len(X)))
    m = XGBQ(7, seed=0, columns=list(X.columns), fixed_rounds=50, max_depth=4, learning_rate=0.1).fit(X, y)
    return m, X.iloc[:800]


@pytest.fixture(scope="module")
def feature_model(feats):
    """a normalised model on the synthetic world's real feature columns (no ids), for the sentence-level tests"""
    P = 7
    cols = [c for c in columns(P) if c not in ("item_id", "store_id", "dept_id", "cat_id")]
    train = feats[feats.date + pd.Timedelta(days=P) <= feats.date.quantile(0.7)]
    m = XGBQ(P, seed=0, columns=cols, normalize=True, floor=1 / 7, fixed_rounds=40, max_depth=3).fit(train, train[f"y_p{P}"])
    sundays = feats[(feats.date > feats.date.quantile(0.7)) & (feats.date.dt.dayofweek == 6)].reset_index(drop=True)
    return m, sundays, P


# ---------- themes ----------
def test_every_model_input_is_in_exactly_one_theme():
    ev = ["Thanksgiving", "Father_s_day"]
    for P in PS:
        for name in columns(P, ev=ev, xs=True):
            assert theme_of(name) in THEMES, name
    with pytest.raises(KeyError):
        theme_of("some_new_feature")                                          # never a silent "other"


def test_series_age_is_not_interpreted_in_planner_sentences(feature_model):
    """age_days sits in the neutral theme; no sentence ever names or interprets it, while global tables label it as not interpreted"""
    assert theme_of("age_days") == "other factors" and "series age" not in THEMES
    assert explain.GLOBAL_LABEL["other factors"] == "series age (not interpreted)"
    m, rows, P = feature_model
    texts = [r[k]["text"] for r in explain_rows(m, rows.iloc[:80], P, 0.9) for k in ("p50", "service")]
    assert texts and not any(re.search(r"\bage\b", t, re.I) for t in texts)
    assert any("remaining-factors group" in t for t in texts), "blind test: the neutral theme should be narrated somewhere in these rows"


def test_theme_contributions_are_additive():
    names = columns(7)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(20, 2, len(names)))
    th, order = theme_contributions(phi, names)
    assert order == list(THEMES) and th.shape == (20, 2, len(THEMES))
    assert np.allclose(th.sum(-1), phi.sum(-1))                               # grouping does not change the total


# ---------- units ----------
def test_ratio_to_units_by_hand():
    assert ratio_to_units([[0.5, -0.25]], [8.0]).tolist() == [[4.0, -2.0]]
    assert ratio_to_units([[1.0, 2.0], [3.0, 4.0]], [2.0, 10.0]).tolist() == [[2.0, 4.0], [30.0, 40.0]]        # one scale per row


# ---------- exactness ----------
def test_sliced_boosters_reproduce_the_sorted_joint_prediction_and_expose_crossing(crossing_model):
    m, X = crossing_model
    unsorted = m._unsorted(X)
    assert np.allclose(np.sort(unsorted, axis=1), m._raw(X), atol=1e-4), "joint predict is the sorted tree sums"
    assert m.crossing_share(X) > 0 and (np.diff(unsorted, axis=1) < 0).any(), "blind test: this fixture must contain crossing rows"


def test_joint_pred_contribs_does_not_reconcile_on_crossing_rows(crossing_model):
    """documents the XGBoost 3.4.1 defect that forces the sliced boosters; if an upgrade fixes it this fails and the workaround can be revisited"""
    m, X = crossing_model
    d = xgb.DMatrix(m._matrix(X, False))
    gap = np.abs(m.model.predict(d, pred_contribs=True).sum(-1) - m.model.predict(d)).max()
    assert gap > 1e-3


def test_additivity_per_quantile_including_crossing_rows(crossing_model):
    m, X = crossing_model
    c = m.contributions(X, ALPHAS)
    raw = m._raw(X)
    close = lambda a, b: np.allclose(a, b, rtol=1e-5, atol=1e-4)                     # float32 trees: relative tolerance
    assert close(c["value"], raw), "bias + sum of contributions must equal the reported quantile for every quantile"
    assert close(c["phi"].sum(-1) + c["bias"], raw)
    crossed = c["source"] != np.arange(len(ALPHAS))[None, :]
    assert crossed.any(), "blind test: some reported quantile must come from another target's trees"
    assert close(c["value"][crossed.any(axis=1)], raw[crossed.any(axis=1)]), "additivity must hold on the crossing rows too"


def test_units_end_to_end_for_a_normalised_model(feature_model):
    m, rows, P = feature_model
    c = m.contributions(rows, [0.5, 0.9])
    d = xgb.DMatrix(m._matrix(rows, False))
    parts = np.stack([b.predict(d, pred_contribs=True) for b in m._single_boosters()])          # ratio space, one booster per target
    scale = m.scale(rows)
    for k in range(2):
        mine = ratio_to_units(parts[c["source"][:, k], np.arange(len(rows)), :], scale)
        assert np.allclose(np.column_stack([c["phi"][:, k], c["bias"][:, k]]), mine, atol=1e-6)
    raw = m._raw(rows)
    assert np.allclose(c["value"], raw[:, [1, 3]], rtol=1e-5, atol=1e-4), "units: baseline + contributions = the quantile in units"


def test_contributions_survive_save_and_load(feature_model, tmp_path):
    import models
    m, rows, P = feature_model
    m.save(tmp_path / "m.json")
    m2 = models.load(tmp_path / "m.json")
    a, b = m.contributions(rows, [0.5, 0.9]), m2.contributions(rows, [0.5, 0.9])
    assert np.array_equal(a["phi"], b["phi"]) and np.array_equal(a["bias"], b["bias"])


# ---------- sentences ----------
def test_sign_to_word_agreement_property():
    rng = np.random.default_rng(2)
    themes = list(THEMES)
    for _ in range(300):
        c = rng.normal(scale=rng.choice([0.03, 0.3, 3.0]), size=len(themes))
        cl = clauses(c, themes)
        chosen = [i for i in np.argsort(-np.abs(c), kind="stable") if abs(c[i]) >= explain.MIN_ABS_UNITS][:explain.TOP]
        assert len(cl) == len(chosen) <= 3
        for (subject, verb, amount), i in zip(cl, chosen):
            assert verb == (VERB_UP if c[i] > 0 else VERB_DOWN), (subject, c[i])
            assert amount == pytest.approx(abs(c[i]))
        text = sentence("P50 forecast", 5.0, 4.0, 7, cl)
        verbs = re.findall(rf"({VERB_UP}|{VERB_DOWN}) it by", text)
        assert verbs == [v for _, v, _ in cl]


def test_tiny_contributions_are_not_narrated():
    assert clauses(np.array([0.04, -0.049, 0.0, 0, 0, 0, 0, 0]), list(THEMES)) == []


def test_price_wording_by_hand():
    assert price_words(0.85, 1.0) == "price is 15% below usual"
    assert price_words(1.10, 1.0) == "price is 10% above usual"
    assert price_words(1.004, 1.0) == "price is about usual"
    assert price_words(1.0, 0.70) == "price is 30% below usual"                  # the planned window price deviates more than today's price
    assert price_words(1.0, 1.0, 0.60) == "price is 40% below usual"             # the planned window minimum deviates most
    assert price_words(1.02, 0.97, 1.0) == "price is 3% below usual"


def test_price_clause_uses_the_price_wording():
    themes = list(THEMES)
    c = np.zeros(len(themes))
    c[themes.index("price vs usual")] = -1.2
    (subject, verb, amount), = clauses(c, themes, price_text="price is 15% below usual")
    assert "price is 15% below usual" in subject and verb == VERB_DOWN and amount == pytest.approx(1.2)


def test_the_absolute_price_level_is_never_described_as_below_usual():
    """the absolute price is an item-type proxy; only the relative-to-usual features may carry the "% below usual" wording"""
    assert theme_of("price") == "price level" and theme_of("price_rel_now") == theme_of("price_min_rel_p7") == "price vs usual"
    themes = list(THEMES)
    c = np.zeros(len(themes))
    c[themes.index("price level")] = 2.0
    (subject, verb, amount), = clauses(c, themes, price_text="price is 15% below usual")
    assert "usual" not in subject and "%" not in subject and verb == VERB_UP


def test_subjects_agree_in_number_with_the_verb():
    """the last word of every subject is singular, so "raises" / "lowers" is grammatical (a plural last word would read "effects raises")"""
    for theme, subject in THEMES.items():
        assert not re.search(r"\b(effects|events|sales|days|stretches)$", subject), (theme, subject)
        assert not re.search(r" and ", subject), (theme, subject)


def test_banned_words_never_appear_and_the_guard_fires(feature_model):
    m, rows, P = feature_model
    texts = [r[k]["text"] for r in explain_rows(m, rows.iloc[:60], P, 0.9) for k in ("p50", "service")]
    assert texts and not any(BANNED.search(t) for t in texts)
    assert all(t.startswith(("P50 forecast", "P90 order-up-to quantile")) for t in texts)
    with pytest.raises(AssertionError):
        sentence("P50 forecast", 3.0, 2.0, 7, [("the promotion", VERB_UP, 1.0)])          # the wording guard itself


# ---------- guards ----------
def test_only_sunday_review_origins_are_explained(feature_model):
    m, rows, P = feature_model
    bad = rows.copy()
    bad.loc[0, "date"] = bad.loc[0, "date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Sunday"):
        explain_rows(m, bad.iloc[:3], P, 0.9)


def test_test_window_origins_are_refused_without_the_flag_and_a_touch_log_reference(feature_model):
    m, rows, P = feature_model
    tw = rows.iloc[:3].copy()
    tw["date"] = TEST_START                                                             # a Sunday review origin of the test window
    with pytest.raises(RuntimeError):
        explain_rows(m, tw, P, 0.9)
    with pytest.raises(ValueError, match="touch-log"):
        explain_rows(m, tw, P, 0.9, allow_test_window=True)
    assert len(explain_rows(m, tw, P, 0.9, allow_test_window=True, touch_log_ref="section 14: final frozen run")) == 3


def test_explanations_are_deterministic_and_consistent(feature_model):
    m, rows, P = feature_model
    a, b = explain_rows(m, rows.iloc[:10], P, 0.95), explain_rows(m, rows.iloc[:10], P, 0.95)
    assert a == b
    for r in a:
        for k in ("p50", "service"):
            e = r[k]
            assert abs(sum(e["themes"].values()) + e["baseline"] - e["value"]) < 1e-3, "themes + baseline = the reported value"
            assert f"{max(e['value'], 0):.1f} units" in e["text"] and f"typical {e['baseline']:.1f}" in e["text"]
