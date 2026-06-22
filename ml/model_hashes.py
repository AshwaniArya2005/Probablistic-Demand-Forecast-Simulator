"""Lists every frozen research model file with the SHA-256 recorded in models/manifest.jsonl when it was saved, re-verifies the file on disk against it, and writes docs/results/model_hashes.md (hashes and
metadata only; no weights). The latest manifest entry per file counts (development models were refit once through the generalised builder, with identical forecasts).
    uv run python ml/model_hashes.py"""
import hashlib
import json
from datetime import date

from config import ROOT

rows = {}
for line in (ROOT / "models" / "manifest.jsonl").read_text().splitlines():
    r = json.loads(line)
    rows[r["file"]] = r
out, bad = [], []
for f, r in sorted(rows.items()):
    p = ROOT / "models" / f
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    ok = h == r["sha256"]
    if not ok:
        bad.append(f)
    out.append(f"| {f} | {r.get('unit', r.get('fold'))} | {r['P']} | {r['role']} | {r['cutoff']} | `{r['commit']}` | `{r['sha256']}` | {'yes' if ok else '**NO**'} |")
md = f"""# Frozen research models: recorded hashes

Checked {date.today()}. Each file below was written by `ml/sim_rows.py` (`save()`), which records its SHA-256 in `models/manifest.jsonl` (git-ignored, with the weights); every file on disk was hashed again and compared.
**{len(rows) - len(bad)} of {len(rows)} match.** The forecast tables that every reported result reads were produced by these fits. The hashes are checked here and by the serving export (`models.lock.json`, verified by the FastAPI service
at start-up); the research pickles themselves are not re-hashed each time a script runs.

| file | unit | horizon | role | cutoff | code commit | SHA-256 | matches |
|---|---|---:|---|---|---|---|---|
""" + "\n".join(out) + "\n"
(ROOT / "docs" / "results" / "model_hashes.md").write_text(md, encoding="utf-8")
print(f"{len(rows) - len(bad)} of {len(rows)} match", bad)
