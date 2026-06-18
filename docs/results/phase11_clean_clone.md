# Clean-clone reproduction (2026-09-21)

A fresh `git clone` of the local repository at commit `8508176` into an empty directory, `uv sync --frozen`, the three raw M5 files copied into `data/raw/`, then:

| step | result |
|---|---|
| `ml/prepare.py`, `ml/features.py` | ran; `panel.parquet` and `features.parquet` **exactly equal** (`DataFrame.equals`) to the working copy's |
| `uv run pytest -q` (fast suite) | 397 passed, 9 deselected |
| `uv run pytest -q -m realdata` | 9 passed |
| `ml/run_baselines.py` (Phase 5) | **failed**: `AttributeError: 'LR' object has no attribute 'model'`. The script iterated over `REGISTRY`, which has held the learned models since Phase 6, and called `predict` on an unfitted one. A real reproducibility defect that only a clean run exposes. Fixed to iterate over the three stateless baselines; the regenerated `phase5_baselines_tuning.csv` and `phase5_baselines_tuning_by_segment.csv` are **identical** to the committed files (only the run-header line of the markdown differs, so the markdown was left as committed). |

**Not reproduced in the clean clone:** the model-fitting stages (Phase 6 onward). Their caches (`data/processed/phase*_cache*.json`) are git-ignored and a cold rebuild is many hours of fits, so
the clean clone stops at the data pipeline, the tests and the baselines. What was reproduced elsewhere: in the working tree the F2 and F4 development forecast tables were rebuilt through the generalised builder and match
the first build exactly (maximum difference 0.0), and the Phase 9 refits reproduce every cached Phase 7 cell value to 1e-9. The gap is stated rather than closed: a full cold reproduction of Phases 6 to 10 has not been done.
