"""FastAPI model service. Kept in the repository and run by docker-compose; NOT part of the deployed public demo (which serves precomputed results only).
/health returns only the status and the model version. /health/model and /quantiles need the X-API-Key header (the Express service or the compose stack holds it)."""
import hmac
import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import serving


class QuantileRequest(BaseModel):
    features: dict[str, float | None] = Field(..., description="every model input column, plus mean_28 when the model is normalised")


def create_app(model_dir=None, lock_path=None, name=None, api_key=None):
    name = name or os.environ.get("MODEL_NAME", "model")
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "models"))
    lock_path = Path(lock_path or os.environ.get("LOCK_FILE", "models.lock.json"))
    api_key = api_key if api_key is not None else os.environ.get("API_KEY", "")
    app = FastAPI(title="Demand forecast model service", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"model": None, "hash_ok": False, "version": None}

    def load():
        state["hash_ok"] = serving.verify(model_dir, lock_path)
        state["model"] = serving.load(model_dir, name) if state["hash_ok"] else None
        try:
            import json
            state["version"] = json.loads(lock_path.read_text(encoding="utf-8")).get("version")
        except (OSError, ValueError):
            state["version"] = None

    load()

    def need_key(x_api_key: str = Header(default="")):
        if not api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="unauthorised")

    @app.get("/health")
    def health():
        ok = state["model"] is not None
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "unavailable", "model_version": state["version"]})

    @app.get("/health/model", dependencies=[Depends(need_key)])
    def health_model():
        return {"loaded": state["model"] is not None, "hash_ok": state["hash_ok"]}

    @app.post("/quantiles", dependencies=[Depends(need_key)])
    def quantiles(req: QuantileRequest):
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not available")
        booster, meta = state["model"]
        missing = [c for c in meta["columns"] if c not in req.features] + ([] if "mean_28" in req.features or not meta.get("normalize") else ["mean_28"])
        if missing:
            raise HTTPException(status_code=422, detail=f"missing inputs: {missing[:5]}")
        q = serving.predict_quantiles(booster, meta, [req.features])[0]
        return {"horizon": meta["P"], "quantiles": {f"q{round(a * 100)}": float(v) for a, v in zip(serving.ALPHAS, q)}}

    return app


app = create_app() if os.environ.get("MODEL_DIR") else None
