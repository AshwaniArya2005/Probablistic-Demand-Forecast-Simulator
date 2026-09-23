# Live per-series model inference for the demo dashboard

Date: 2026-09-23
Status: approved, not yet implemented

## Context

The deployed demo (Phase 11, `server/`, `web/`) serves **precomputed** results only:
`quantiles` table in Postgres holds final model outputs (q10/q50/q80/q90/q95/q99, zero_run)
per (series, review_date, horizon), and `/api/whatif` computes the order-quantity what-if
from that stored row. No request ever reaches a live model.

This adds a second, parallel path: a real round trip through the trained model
(`api/app.py`, FastAPI, already coded) for a real M5 series and review date, proving the
serving pipeline actually works end to end — not just replaying a number that was computed
offline months ago.

**Scope decision, already made and not reopened by this spec:** serving this per-series data
(and features derived from it) publicly requires the M5 competition's data-use terms to
permit it. The repository owner has confirmed this is fine. This spec does not re-litigate
that; it only extends the *technical* pipeline that the owner's decision unblocks. It reuses
the same gate `server/scripts/load.js` already enforces (`ALLOW_PUBLIC_QUANTILES=confirmed-kaggle-rules`).

**Not in scope:** a UI for typing in arbitrary/synthetic feature values, a P14 model (only
`v4_P10` is exported to `models/serving/`), computing features from raw sales history at
request time (features are precomputed and stored, same as quantiles are today).

## Architecture

```
                         (existing, unchanged)
Browser ──fetch──▶ Express /api/whatif ──▶ Postgres.quantiles ──▶ precomputed answer

                         (new, this spec)
Browser ──click "Verify live"──▶ Express /api/whatif/live
                                     │
                                     ├─▶ Postgres.features  (stored feature row for series+date+horizon)
                                     │
                                     └─▶ Render: FastAPI model service  ──▶ v4_P10.ubj (XGBoost)
                                             (X-API-Key header, held server-side only)
                                             returns live quantiles
```

Hosting split:
- **Vercel**: static site (`web/`) and the Express API (`server/`), both as Vercel projects.
- **Render**: the Python/XGBoost model service (`api/`), as a Docker web service — chosen over
  Vercel serverless because it needs to load the model once and stay warm, with no
  serverless bundle-size ceiling to fight.

## A. Data pipeline

Extend `ml/build_demo_data.py` with a `features()` function, mirroring the existing
`quantiles()`:

- Same universe as `quantiles.csv`: the rows already in `features.parquet` for the 300
  series × 25 test-window review dates × horizon 10 that `quantiles()` already selects.
- For each row, take exactly `ml/features.py`'s `columns(P=10)` list (the model's real input
  contract) plus `mean_28` (needed when the model is normalised, same as
  `QuantileRequest` in `api/app.py` expects).
- Write `demo_private/features.csv` (git-ignored, same as `quantiles.csv` — "do not publish"
  header comment carried over).

## B. Schema

```sql
CREATE TABLE IF NOT EXISTS features (
  series text NOT NULL, review_date date NOT NULL, horizon int NOT NULL,
  payload jsonb NOT NULL,
  PRIMARY KEY (series, review_date, horizon)
);
```

A `jsonb` payload (like `scenarios.payload`) rather than one column per feature: the feature
set is defined by `ml/features.py::columns()`, and a schema of ~35 typed columns would need a
migration every time that list changes. `quantiles` uses fixed columns because its 7 outputs
are stable; `features` doesn't have that stability guarantee.

`server/scripts/load.js` loads `features.csv` into this table under the same non-local guard
already covering `--quantiles` (`ALLOW_PUBLIC_QUANTILES=confirmed-kaggle-rules`) — one flag
governs both, since they're the same data-terms question.

## C. Model service (Render)

- Commit `models/serving/` (`models.lock.json`, `v4_P10.meta.json`, `v4_P10.ubj` — 24MB total)
  to the repository. `api/serving.py`'s existing hash verification against `models.lock.json`
  is the integrity check; nothing new needed there.
- `api/Dockerfile` already builds correctly as-is (`COPY api/serving.py api/app.py`, installs
  fastapi/uvicorn/xgboost/numpy) — it just needs the models baked into the image instead of
  the docker-compose bind mount, since Render can't mount a local host path. Add
  `COPY models/serving /models` to the Dockerfile; drop the `MODEL_DIR=/models` env default
  (already set) since it now matches a real in-image path.
- New Render Web Service (Docker), env var `API_KEY` = a freshly generated random secret
  (not `AshwaniArya` or anything memorable — this one is never meant to be typed by a human).
- `/health` (no key) for Render's own health checks; `/health/model` and `/quantiles` require
  `X-API-Key`, exactly as already coded.

## D. Express API (Vercel)

- Wrap the existing `server/src/app.js` `createApp()` as a Vercel serverless function:
  one `api/index.js` (Vercel's convention, distinct from the repo's `api/` FastAPI folder —
  naming collision to watch for, likely lives at `server/api/index.js` with a `vercel.json`
  rewrite) that constructs the app once per cold start and forwards `(req, res)`.
- Postgres pool trimmed for serverless: `max: 1` per function instance, relying on Supabase's
  transaction pooler (already what's configured) to multiplex real connections — a serverless
  function holding `max: 4` idle connections per cold start doesn't fit the pooler model.
- New config passed into `createApp()`: `modelApiUrl`, `modelApiKey` (from Vercel env vars,
  never sent to the browser).
- New route, same validation/error style as the existing `quantileRow`/`/api/whatif`:

  ```js
  async function featureRow(query) { /* same shape as quantileRow, SELECT payload FROM features ... */ }

  app.get('/api/whatif/live', async (req, res, next) => {
    try {
      const alpha = parseAlpha(req.query.alpha);
      const position = parsePosition(req.query.position);
      if (alpha === null || position === null) return res.status(400).json({ error: 'bad request' });
      const f = await featureRow(req.query);
      if (f.status) return res.status(f.status).json({ error: f.status === 400 ? 'bad request' : 'not found' });
      const r = await fetchWithTimeout(`${modelApiUrl}/quantiles`, {
        method: 'POST', headers: { 'X-API-Key': modelApiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: f.payload }),
      }, 65_000); // Render free tier can be cold; mirror the dashboard's existing ~70s wake budget
      if (!r.ok) return res.status(503).json({ error: 'model service unavailable' });
      const body = await r.json();
      const level = body.quantiles[`q${Math.round(alpha * 100)}`];
      const orderUpTo = ceilUnits(level);
      return res.json({
        series: req.query.series, date: req.query.date, horizon: f.h, alpha, quantile: level,
        order_up_to: orderUpTo, position, order_quantity: Math.max(0, orderUpTo - position),
      });
    } catch (e) { return next(e); } // same rule as every other handler: never echo error details
  });
  ```

- `/api/whatif` (precomputed) is untouched — this is purely additive.

## E. Static site (Vercel)

Deploy `web/` as its own Vercel project (or the same project's static output, either is fine —
implementation plan decides). `web/config.js` updated to the Vercel API project's URL.

## F. Dashboard UI

In `renderWhatIf()` (`web/app.js`), after a successful `/api/whatif` result renders, add a
"Verify live" button. On click: `getJSON('${CFG.API_BASE}/api/whatif/live?...', 65000)`,
render its quantile next to the stored one with a one-line label ("live model service: should
match the stored value — this is the same model, called fresh"). A mismatch is not expected
and not specially handled beyond showing both numbers; if it happens that's a real bug to
investigate, not a UI state to design around.

Loading state while waiting: reuse the existing "waking the service" copy pattern already in
`boot()`, since the Render free tier has the identical spin-down behavior as the main API.

## G. Testing

- `server/test/app.test.js`: new cases for `/api/whatif/live` — bad params (400), missing
  feature row (404), model service unreachable (503, mocked fetch), success path (mocked
  fetch returning a fixed quantile payload, assert the same `ceilUnits`/order-quantity math
  as the precomputed path).
- Python side: `tests/test_models_contract.py` (existing) already checks the reference model
  contract; confirm it covers `v4_P10` specifically. If a request built from a real
  `features.csv` row round-trips through `api/serving.py::predict_quantiles` to the same
  value already stored in `quantiles.csv` for that row, that's the strongest correctness
  check available (live and precomputed must agree, since they're the same model) — add this
  as one test if not already covered by something equivalent.
- No changes to `ml/`'s existing test suite otherwise; frozen v1 models untouched (this is a
  new serving path for an existing model, not a new model).

## Error handling summary

- Malformed query params → 400, same regex-based validation already used for `/api/whatif`.
- Series/date/horizon combination not in the precomputed `features` table → 404 (this path
  only covers the same 300×25×1 universe as `quantiles`, not arbitrary series/dates).
- Model service unreachable, cold, or returns non-2xx → 503, generic message (no upstream
  error details echoed — same rule as the existing global error handler).
- Model service key misconfigured → indistinguishable from "unreachable" to the client (503);
  real cause visible in Vercel/Render logs only, never in the response body.
