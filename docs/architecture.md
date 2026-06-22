# Architecture

Two things live in this repository: the **research pipeline** (Python, run offline) and a **demo** that serves its precomputed results. The public demo runs no model.

## 1. Research pipeline

```
data/raw (M5 csv, not in the repo)
   │  ml/prepare.py      clean, closed-day rule (25 Dec), velocity segments, 100-item sample -> panel.parquet
   │  ml/features.py     leakage-audited features and targets (docs/features.md) -> features.parquet
   ▼
rolling-origin evaluation (ml/folds.py, ml/versions.py)
   │  tuning folds F1-F4 (12 Sunday origins each, all ending before 2015-08-30)   -> model selection
   │  model versions v0-v4 with embargoed fit sets (fit end = cutoff - 84 days)    -> test window
   ▼
models (ml/models.py, ml/learned.py, ml/quantiles.py, ml/conformal.py)
   │  baselines: naive, MA-28, seasonal naive
   │  point: Linear Regression, Random Forest, XGBoost (mean)
   │  quantile: one XGBoost multi-quantile model per horizon (0.10 ... 0.99), sorted, normalised target
   ▼
policy layer (ml/policy.py)          order-up-to levels: naive, point (normal sigma), quantile, post-hoc empirical
   ▼
simulator (ml/simulator.py, ml/sim_policies.py, ml/sim_run.py)     periodic review, lost sales, integer units
   ▼
reports (docs/results/*.md, *.csv)   every run leaves a file; the decision log is docs/design.md
```

Main modules and what they promise:

| Module | Role | Guard |
|---|---|---|
| `folds.py`, `versions.py` | folds, model versions, embargo and calibration windows, review dates | `assert_tuning_only` refuses any target window that reaches the test window; `touch_logged` refuses a test-window run without its log row |
| `features.py` | features and targets | brute-force reference and perturbation tests: no future sales or prices in any input |
| `learned.py` | LR / RF / XGBoost / XGBoost-quantile | shared contract tests (shape, determinism, causality, save/load) |
| `simulator.py` | the replay engine (pure numpy) | hand-computed, timing, conservation, integer, limit-case, leakage tests and an independent reference implementation |
| `sim_run.py` | pools runs, bootstrap over items, reports | frozen naive grid, one primary run, refuses to overwrite a result file |
| `explain.py` | exact per-quantile contributions, planner sentences | additivity, sign-to-word and wording tests; tuning-only unless a touch-log reference is given |

Process infrastructure: the decision log and pre-registered rules (`docs/design.md`), the test-window touch log (section 14 of it), a phase gate (`docs/phase_status.json`, `ml/phase_gate.py`), a recorded test log
(`docs/results/test_log.md`), and CI (`.github/workflows/tests.yml`: the fast Python suite and the Node suite).

## 2. The demo (built and tested locally, not deployed)

```
result files (docs/results)  ──ml/build_demo_data.py──►  web/snapshot.json   (aggregates only, committed)
                                                      └►  demo_private/quantiles.csv  (per-series tables, git-ignored)

Postgres  ◄── server/scripts/load.js ── (scenarios always; quantile tables only into a local database)
   ▲
Express (server/)  /health   /health/deep   /api/status   /api/scenarios[/:id]   /api/quantiles   /api/whatif
   ▲
static page (web/)  renders the snapshot at once, swaps to live data when the API answers
```

- **No live model.** The order-quantity what-if is `max(0, ceil(q) - inventory position)`, computed in Express from a stored quantile, with the stockout-risk band (a band, not a probability) and its two qualifiers.
- **Health (availability only).** `/health` is liveness. `/health/deep` runs `SELECT 1` and checks the precomputed-data version; on failure it answers `503` with a short reason and nothing else. `/api/status` records the last deep check for the footer.
  One external monitor on `/health/deep` every five minutes is the intended setup (not yet created). Drift monitoring is out of scope.
- **Abuse limits.** Rate limiting on every route, a stricter one on `/health/deep`, generic error bodies, input validation on every query parameter.
- **Publication gate.** Per-series quantile tables are derived from M5 sales; the loader refuses a non-local database until the data-use terms are confirmed.
- **FastAPI model service** (`api/`, `docker-compose.yml`): kept for local use, not deployed. It loads only native XGBoost files (`ml/export_models.py`) whose SHA-256 matches a tracked `models.lock.json`, refuses to serve on a mismatch, and
  reports health with only a status and a model version; hash checks sit behind an API-key route.

## 3. Deployment plan and its constraints

Frontend on a static host, Express on a free Render web service, Postgres on Supabase, one uptime monitor. Free-tier facts read on 2026-09-21 (re-check before deploying): Render spins a free service down after 15 minutes idle and gives 750 instance hours a month
(a five-minute health check keeps one service awake, about 744 hours); Supabase pauses a free project after a week idle (the monitor's `SELECT 1` prevents that); Hugging Face Docker Spaces need a paid plan, which is why FastAPI is not part of the public demo.
Nothing has been deployed and no monitor exists yet.
