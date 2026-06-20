"""Exports a research quantile model (pickled `XGBQ`, models/*.json written by ml/sim_rows.py) to the native serving format: `<name>.ubj` (XGBoost booster) + `<name>.meta.json`
(horizon, alphas, input columns, normalisation flag, scale floor), and writes `models.lock.json` with SHA-256 hashes. The research pickle is loaded once, locally, from a file this
repository wrote (never untrusted input). Usage: uv run python ml/export_models.py <research model json> <out dir> <name> <version>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
import serving  # noqa: E402
from learned import XGBQ  # noqa: E402


def export(research_json, out_dir, name, version):
    m = XGBQ.load(research_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m.model.save_model(str(out / f"{name}.ubj"))
    meta = dict(P=m.P, alphas=list(m.ALPHAS), columns=m.columns, normalize=bool(m.normalize), floor=float(m.floor))
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return serving.write_lock(out, [f"{name}.ubj", f"{name}.meta.json"], version, out / "models.lock.json")


if __name__ == "__main__":
    print(export(*sys.argv[1:5]))
