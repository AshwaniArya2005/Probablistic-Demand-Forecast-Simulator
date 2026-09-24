"""FastAPI model service. Kept in the repository and run by docker-compose; NOT part of the deployed public demo (which serves precomputed results only).
/health returns only the status and the model version. /health/model and /quantiles need the X-API-Key header (the Express service or the compose stack holds it)."""
import hmac
import json
import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import serving


class QuantileRequest(BaseModel):
    features: dict[str, float | None] = Field(..., description="every model input column, plus mean_28 when the model is normalised")
    model: str | None = Field(default=None, description="which loaded model to use; defaults to the service's default model")


def create_app(model_dir=None, lock_path=None, name=None, api_key=None):
    name = name or os.environ.get("MODEL_NAME") or None
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "models"))
    lock_path = Path(lock_path or os.environ.get("LOCK_FILE", "models.lock.json"))
    api_key = api_key if api_key is not None else os.environ.get("API_KEY", "")
    app = FastAPI(title="Demand forecast model service", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"models": {}, "hash_ok": False, "version": None, "default": name}

    def discover_names():
        """every `<name>.meta.json` listed in the lock file names a loadable model"""
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            return sorted(n[: -len(".meta.json")] for n in lock["files"] if n.endswith(".meta.json"))
        except (OSError, KeyError, ValueError):
            return [name] if name else []

    def load():
        state["hash_ok"] = serving.verify(model_dir, lock_path)
        state["models"] = {n: serving.load(model_dir, n) for n in discover_names()} if state["hash_ok"] else {}
        if not state["default"] and state["models"]:
            state["default"] = sorted(state["models"])[0]
        try:
            state["version"] = json.loads(lock_path.read_text(encoding="utf-8")).get("version")
        except (OSError, ValueError):
            state["version"] = None

    load()

    def need_key(x_api_key: str = Header(default="")):
        if not api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="unauthorised")

    @app.get("/health")
    def health():
        ok = bool(state["models"])
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "unavailable", "model_version": state["version"]})

    @app.get("/health/model", dependencies=[Depends(need_key)])
    def health_model():
        loaded = (state["default"] in state["models"]) if state["default"] else bool(state["models"])
        return {"loaded": loaded, "hash_ok": state["hash_ok"]}

    @app.post("/quantiles", dependencies=[Depends(need_key)])
    def quantiles(req: QuantileRequest):
        chosen = req.model or state["default"]
        if chosen is None or chosen not in state["models"]:
            raise HTTPException(status_code=503, detail="model not available")
        booster, meta = state["models"][chosen]
        missing = [c for c in meta["columns"] if c not in req.features] + ([] if "mean_28" in req.features or not meta.get("normalize") else ["mean_28"])
        if missing:
            raise HTTPException(status_code=422, detail=f"missing inputs: {missing[:5]}")
        q = serving.predict_quantiles(booster, meta, [req.features])[0]
        return {"horizon": meta["P"], "quantiles": {f"q{round(a * 100)}": float(v) for a, v in zip(serving.ALPHAS, q)}}

    return app


app = create_app() if os.environ.get("MODEL_DIR") else None
