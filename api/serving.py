"""Model serving pieces shared by the FastAPI service and the export test: hash-verified lock file, native XGBoost booster + meta.json, quantile prediction.
No pickles: the service only ever loads native XGBoost files whose SHA-256 matches models.lock.json."""
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

ALPHAS = (0.10, 0.50, 0.80, 0.90, 0.95, 0.99)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_lock(model_dir, names, version, out):
    """lock file: {version, files: {file name: sha256}}. Merges into any existing lock at `out` (so exporting
    several model versions into the same directory accumulates hashes instead of each call wiping the last)."""
    existing = {}
    try:
        existing = json.loads(Path(out).read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        pass
    lock = {"version": version, "files": {**existing, **{n: sha256(Path(model_dir) / n) for n in sorted(names)}}}
    Path(out).write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return lock


def verify(model_dir, lock_path):
    """True only if every file listed in the lock exists in model_dir with the locked hash"""
    try:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        return bool(lock["files"]) and all((Path(model_dir) / n).is_file() and sha256(Path(model_dir) / n) == h for n, h in lock["files"].items())
    except (OSError, KeyError, ValueError):
        return False


def load(model_dir, name):
    """(booster, meta) for `<name>.ubj` and `<name>.meta.json`; caller must have verified the lock first"""
    booster = xgb.Booster()
    booster.load_model(str(Path(model_dir) / f"{name}.ubj"))
    meta = json.loads((Path(model_dir) / f"{name}.meta.json").read_text(encoding="utf-8"))
    return booster, meta


def predict_quantiles(booster, meta, rows):
    """rows: list of dicts holding every column of meta['columns'] (NaN allowed as None). Returns an (n, 6) array of quantiles for ALPHAS, in units:
    joint booster output, times the row scale max(mean_28, floor) * P when the model was trained on the normalised target, clipped at 0 and sorted."""
    cols = meta["columns"]
    X = np.array([[np.nan if r.get(c) is None else r[c] for c in cols] for r in rows], dtype="float32")
    q = np.asarray(booster.predict(xgb.DMatrix(X, feature_names=None)), "float64").reshape(len(rows), len(ALPHAS))
    if meta.get("normalize"):
        m28 = np.array([r["mean_28"] for r in rows], float)
        q = q * (np.fmax(m28, meta["floor"]) * meta["P"])[:, None]
    return np.sort(np.maximum(q, 0.0), axis=1)
