"""The FastAPI model service and the export/lock machinery (docs/design.md, Phase 11 decisions). Synthetic tiny models only; nothing here asserts accuracy on project data.
The service is kept in the repository and run by docker-compose; the deployed public demo serves precomputed results and does not use it."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import serving
from app import create_app
from export_models import export
from learned import XGBQ

KEY = "test-key"
H = {"X-API-Key": KEY}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"mean_28": rng.uniform(0.3, 6, n), "x": rng.normal(size=n)})
    X["y_p10"] = rng.poisson(X.mean_28 * 10 * (1 + 0.2 * X.x.clip(-1, 1))).astype(float)
    return X


@pytest.fixture(scope="module")
def exported(tmp_path_factory, frame):
    d = tmp_path_factory.mktemp("models")
    m = XGBQ(10, seed=0, columns=["mean_28", "x"], normalize=True, floor=1 / 7, fixed_rounds=40)
    m.fit(frame, frame["y_p10"])
    m.save(d / "research.json")
    lock = export(d / "research.json", d / "serving", "model", "v1-test")
    return d / "serving", m, lock


def client(exported):
    model_dir, _, _ = exported
    return TestClient(create_app(model_dir, model_dir / "models.lock.json", "model", KEY))


def test_health_returns_only_status_and_model_version(exported):
    r = client(exported).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "v1-test"}


def test_no_health_response_leaks_hashes_paths_or_configuration(exported):
    model_dir, _, lock = exported
    c = client(exported)
    texts = [c.get("/health").text, c.get("/health/model", headers=H).text]
    for t in texts:
        for banned in [*lock["files"].values(), str(model_dir), "models.lock", KEY, ".ubj", "sha256", "columns", "floor"]:
            assert banned not in t, banned


def test_protected_routes_need_the_api_key(exported):
    c = client(exported)
    for method, path in (("get", "/health/model"), ("post", "/quantiles")):
        assert getattr(c, method)(path).status_code == 401
        assert getattr(c, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/health/model", headers=H).json() == {"loaded": True, "hash_ok": True}
    # a service started without any key refuses everything protected
    model_dir, _, _ = exported
    assert TestClient(create_app(model_dir, model_dir / "models.lock.json", "model", "")).get("/health/model", headers={"X-API-Key": ""}).status_code == 401


def test_a_hash_mismatch_is_reported_and_the_model_is_not_loaded(exported, tmp_path):
    model_dir, _, _ = exported
    for f in model_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    ubj = tmp_path / "model.ubj"
    ubj.write_bytes(ubj.read_bytes() + b"tamper")
    c = TestClient(create_app(tmp_path, tmp_path / "models.lock.json", "model", KEY))
    assert c.get("/health").status_code == 503 and c.get("/health").json()["status"] == "unavailable"
    assert c.get("/health/model", headers=H).json() == {"loaded": False, "hash_ok": False}
    assert c.post("/quantiles", headers=H, json={"features": {"mean_28": 1.0, "x": 0.0}}).status_code == 503


def test_verify_rejects_missing_files_wrong_hashes_and_an_empty_lock(exported, tmp_path):
    model_dir, _, _ = exported
    assert serving.verify(model_dir, model_dir / "models.lock.json")
    assert not serving.verify(tmp_path, model_dir / "models.lock.json")                 # files absent
    assert not serving.verify(model_dir, tmp_path / "nope.json")                        # lock absent
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "x", "files": {}}))
    assert not serving.verify(model_dir, bad)                                           # an empty lock proves nothing
    bad.write_text(json.dumps({"version": "x", "files": {"model.ubj": "0" * 64}}))
    assert not serving.verify(model_dir, bad)


def test_exported_model_reproduces_the_research_model(exported, frame):
    model_dir, research, _ = exported
    booster, meta = serving.load(model_dir, "model")
    rows = frame.iloc[:50]
    got = serving.predict_quantiles(booster, meta, rows[meta["columns"]].to_dict("records"))
    want = research.predict_quantiles(rows, list(serving.ALPHAS))
    assert np.abs(got - want).max() < 1e-4
    assert meta["P"] == 10 and meta["normalize"] and meta["columns"] == ["mean_28", "x"] and meta["alphas"] == list(serving.ALPHAS)


def test_quantiles_endpoint_is_sorted_non_negative_and_rejects_missing_inputs(exported):
    c = client(exported)
    a = c.post("/quantiles", headers=H, json={"features": {"mean_28": 2.0, "x": 0.3}}).json()
    vals = [a["quantiles"][k] for k in ("q10", "q50", "q80", "q90", "q95", "q99")]
    assert a["horizon"] == 10 and vals == sorted(vals) and min(vals) >= 0
    assert c.post("/quantiles", headers=H, json={"features": {"x": 0.3}}).status_code == 422


def test_the_normalised_output_is_multiplied_by_max_mean28_floor_times_P(exported):
    """mean_28 = 0.01 is below the floor 1/7, so the scale is (1/7) * 10; the served quantiles are the raw booster output times it, clipped and sorted"""
    import xgboost as xgb
    model_dir, _, _ = exported
    booster, meta = serving.load(model_dir, "model")
    got = serving.predict_quantiles(booster, meta, [{"mean_28": 0.01, "x": 0.2}])
    raw = booster.predict(xgb.DMatrix(np.array([[0.01, 0.2]], "float32"))).reshape(1, 6)
    assert meta["floor"] == pytest.approx(1 / 7)
    assert got == pytest.approx(np.sort(np.maximum(raw * (1 / 7) * 10, 0), axis=1), rel=1e-5)
