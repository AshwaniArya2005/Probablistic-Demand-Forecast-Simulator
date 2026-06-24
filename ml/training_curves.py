"""Infrastructure only (design.md, 2026-09-22 pre-registration item 2): writes a fitted XGB/XGBoost-quantile model's train/validation curve (populated in
`model.eval_history_` whenever the fit used a date-based early-stopping split) to a small JSON file. Not called by anything that runs today; a future
legitimately-refit version (e.g. a pre-registered v2) calls this next to its other outputs. Never touches the frozen v1 models."""
import json
from pathlib import Path


def save(model, path):
    """model: a fitted XGB or XGBQ instance. Writes {"metric": <name>, "n_rounds": <best round count>, "train": [...], "val": [...]} or a
    {"available": False} placeholder if the model had no date column / no early-stopping split (e.g. too few origins). Returns the dict written."""
    hist = getattr(model, "eval_history_", None)
    if not hist:
        out = {"available": False}
    else:
        metric = next(iter(hist["train"]))
        out = {"available": True, "metric": metric, "n_rounds": int(model.n_rounds_), "hit_round_cap": bool(model.hit_cap_),
               "train": hist["train"][metric], "val": hist["val"][metric]}
    Path(path).write_text(json.dumps(out), encoding="utf-8")
    return out
