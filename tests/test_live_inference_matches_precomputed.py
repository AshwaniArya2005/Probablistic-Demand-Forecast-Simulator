"""Live inference (api/serving.py, called with a stored features.csv row) must reproduce the value already
stored in quantiles.csv for the same (series, date, horizon) — they're the same v4 model, so they must agree.
Needs demo_private/features.csv and demo_private/quantiles.csv (git-ignored, local pipeline output); skipped otherwise."""
import json
from pathlib import Path

import pandas as pd
import pytest

import serving
from config import ROOT

FEATURES = ROOT / "demo_private" / "features.csv"
QUANTILES = ROOT / "demo_private" / "quantiles.csv"


@pytest.mark.realdata
@pytest.mark.skipif(not (FEATURES.exists() and QUANTILES.exists()), reason="demo_private/*.csv not built locally")
def test_live_model_output_matches_the_stored_quantile_for_the_same_row():
    feats = pd.read_csv(FEATURES)
    quant = pd.read_csv(QUANTILES)
    merged = feats.merge(quant, on=["series", "review_date", "horizon"], how="inner")
    assert len(merged) > 0, "no overlap between features.csv and quantiles.csv rows — check versions.use_dates(4) filtering"
    booster, meta = serving.load(ROOT / "models" / "serving", "v4_P10")
    sample = merged.sample(n=min(20, len(merged)), random_state=0)
    for _, row in sample.iterrows():
        got = serving.predict_quantiles(booster, meta, [json.loads(row["payload"])])[0]
        want = [row.q10, row.q50, row.q80, row.q90, row.q95, row.q99]
        assert list(got) == pytest.approx(want, abs=1e-3), (row["series"], row["review_date"])
