"""Real-data checks of the Phase 7 runner's feature sets (importing the runner loads the feature table, so this is marked realdata)."""
import pytest

pytestmark = pytest.mark.realdata


def test_feature_sets_by_variant_and_arm():
    import tune_quantile as tq
    from features import STATIC
    P = 7
    raw, norm = tq.cols_for(P, "raw", "base"), tq.cols_for(P, "norm", "base")
    assert not set(STATIC) & set(raw) and not set(STATIC) & set(norm), "ids are out of both base sets (Phase 6 outcome)"
    assert f"sum_364_{P}" not in raw and f"sum_364_{P}" in norm, "year-ago follows the variant: kept only for the normalised one"
    assert len(tq.EV) == 30
    assert len(tq.cols_for(P, "raw", "ev")) == len(raw) + 30 and len(tq.cols_for(P, "raw", "xs")) == len(raw) + 2
    assert set(tq.cols_for(P, "raw", "ev+xs")) == set(raw) | {f"ev_{n}_p{P}" for n in tq.EV} | {"other_zero_run_91", "other_zero_28"}
    assert set(tq.cols_for(P, "raw", "yearago_flip")) - set(raw) == {f"sum_364_{P}"}
    assert set(norm) - set(tq.cols_for(P, "norm", "yearago_flip")) == {f"sum_364_{P}"}
    assert set(raw) - set(tq.cols_for(P, "raw", "nofutprice")) == {f"price_mean_rel_p{P}", f"price_min_rel_p{P}"}
    assert set(tq.cols_for(P, "raw", "ids_all")) - set(raw) == set(STATIC)
    assert set(tq.cols_for(P, "raw", "ids_noitem")) - set(raw) == set(STATIC) - {"item_id"}
