# Live Per-Series Model Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, parallel "verify live" path through the real FastAPI/XGBoost model service for a real M5 series+date, sitting alongside the existing precomputed `/api/whatif` (which is untouched).

**Architecture:** A new `features` table (Postgres) stores the ~35 model input columns per (series, review_date, horizon=10), for exactly the test dates the deployed `v4` model version actually served. A new Express endpoint fetches that row and proxies it to the FastAPI model service (Render, Docker, holds the model warm in memory) with a server-held API key. The dashboard adds a "Verify live" button next to the existing precomputed what-if result.

**Tech Stack:** Node.js (Express, `pg`, `node:test`), Python (pandas, FastAPI — already coded, not modified), Postgres (Supabase-hosted), Render (Docker web service), Vercel (static site + serverless Express).

**Spec:** `docs/superpowers/specs/2026-09-23-live-per-series-inference-design.md`

## Global Constraints

- Live inference only covers 300 series × `versions.use_dates(4)` (the tail slice of the 25-date test window served by the `v4` model), horizon 10 — **not** all 25 dates like `quantiles.csv`. A date outside this slice returns 404 from the live endpoint; this is correct, not a bug.
- `demo_private/features.csv` is git-ignored, same as `demo_private/quantiles.csv` — never commit it.
- Loading `features` into a non-local database requires `ALLOW_PUBLIC_QUANTILES=confirmed-kaggle-rules` — same gate as `--quantiles`, already in `server/scripts/load.js`.
- Every error response is generic (`{"error": "..."}`), never echoes upstream details — same rule the existing global error handler in `server/src/app.js` already follows.
- `/api/whatif` (precomputed) is not modified. This plan is purely additive to `server/src/app.js`.
- Model API key is never sent to the browser — held server-side only, read from an env var.

---

### Task 1: `features` table in the schema

**Files:**
- Modify: `db/schema.sql`

**Interfaces:**
- Produces: a `features` table with columns `(series text, review_date date, horizon int, payload jsonb)`, primary key `(series, review_date, horizon)` — the shape Task 2's loader and Task 5's Express query both depend on.

- [ ] **Step 1: Add the table**

Append to `db/schema.sql`:

```sql
-- per-series model input features for live inference: covers only the v4-served slice of test dates (a subset of `quantiles`), same publication gate
CREATE TABLE IF NOT EXISTS features (
  series text NOT NULL, review_date date NOT NULL, horizon int NOT NULL,
  payload jsonb NOT NULL,
  PRIMARY KEY (series, review_date, horizon)
);
```

- [ ] **Step 2: Verify it's valid SQL against the local stack**

Run (with the local Docker Postgres from earlier in this project already up — `docker compose up -d db` if not):
```
docker compose run --rm -e DATABASE_URL=postgres://demo:demo@db:5432/demo server node -e "require('pg').Pool ? require('/app/server/../db/schema.sql') : 0" 2>&1 || true
```
Simpler: just re-run the existing loader, which executes the whole `schema.sql` file as its first step —
```
docker compose run --rm -e DATABASE_URL=postgres://demo:demo@db:5432/demo server node scripts/load.js
```
Expected: `loaded 5 scenarios; data version ...` (same as before — this step only proves the new `CREATE TABLE IF NOT EXISTS features` line doesn't break the existing load).

- [ ] **Step 3: Commit**

```bash
git add db/schema.sql
git commit -m "Add features table for live per-series model inference"
```

---

### Task 2: Feature export pipeline

**Files:**
- Modify: `ml/build_demo_data.py`
- Test: manual verification (no fixture data available for this in CI — same as the existing `quantiles()` function, which also has no automated test; see `tests/test_demo_data.py`, which only covers `build()`)

**Interfaces:**
- Consumes: `data/processed/features.parquet` (built by `ml/features.py`, requires real M5 data — developer must have already run the full pipeline locally), `models/serving/v4_P10.meta.json` (Task 4 commits this; if Task 4 hasn't run yet, read it from the local `models/serving/` directory, which already exists on disk even before it's committed to git), `ml/versions.py::use_dates`.
- Produces: `demo_private/features.csv` with columns `series, review_date, horizon, <every column in v4_P10.meta.json's "columns" list>`.

- [ ] **Step 1: Write the `features()` function**

Add to `ml/build_demo_data.py`, after the existing `quantiles()` function:

```python
def features():
    import versions
    meta = json.loads((ROOT / "models" / "serving" / "v4_P10.meta.json").read_text(encoding="utf-8"))
    cols = meta["columns"]
    dates = versions.use_dates(4)
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["id", "date", *cols])
    feats = feats[feats.date.isin(dates)]
    out = pd.DataFrame({
        "series": feats["id"], "review_date": feats["date"].dt.strftime("%Y-%m-%d"), "horizon": 10,
        "payload": feats[cols].apply(lambda r: json.dumps({c: (None if pd.isna(v) else float(v)) for c, v in r.items()}), axis=1),
    })
    (ROOT / "demo_private").mkdir(exist_ok=True)
    out.to_csv(ROOT / "demo_private" / "features.csv", index=False)
    print("wrote demo_private/features.csv:", len(out), "rows (git-ignored; do not publish)")
```

- [ ] **Step 2: Wire up the CLI dispatch**

Modify the bottom of `ml/build_demo_data.py`:

```python
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    {"quantiles": quantiles, "features": features}.get(arg, build)()
```

- [ ] **Step 3: Run it and hand-verify the output**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator"
uv run python ml/build_demo_data.py features
```

Expected: `wrote demo_private/features.csv: <N> rows`, where N = 300 series × (number of dates in `versions.use_dates(4)`). Then hand-check:

```bash
uv run python -c "
import pandas as pd, json
d = pd.read_csv('demo_private/features.csv')
print(d.series.nunique(), 'series,', d.review_date.nunique(), 'dates')
row = json.loads(d.payload.iloc[0])
meta = json.loads(open('models/serving/v4_P10.meta.json').read())
assert set(row.keys()) == set(meta['columns']), sorted(set(row) ^ set(meta['columns']))
print('columns match v4_P10.meta.json: OK')
"
```

Expected: `300 series, <k> dates` (k should be noticeably smaller than 25 — it's only v4's slice) and `columns match v4_P10.meta.json: OK`.

- [ ] **Step 4: Confirm `demo_private/` is git-ignored**

```bash
git check-ignore demo_private/features.csv
```

Expected: prints the path (confirms it's ignored). If this prints nothing, **stop** — check `.gitignore` for a `demo_private/` entry before proceeding (it should already be there, since `quantiles.csv` already lives there safely).

- [ ] **Step 5: Commit the code change only**

```bash
git add ml/build_demo_data.py
git commit -m "Add features() export for live per-series model inference"
```

---

### Task 3: Loader support for `features.csv`

**Files:**
- Modify: `server/scripts/load.js`

**Interfaces:**
- Consumes: `demo_private/features.csv` (Task 2's output), the `features` table (Task 1).
- Produces: `--quantiles` now also loads `features` (folded into the same flag and the same Kaggle-terms gate — they're the same data-publication question, no reason to split it into a second flag).

- [ ] **Step 1: Extend the loader**

In `server/scripts/load.js`, inside the `if (withQuantiles) { ... }` block (after the existing `quantiles` table load, before the closing brace), add:

```js
  if (withQuantiles) {
    const fpath = path.join(root, 'demo_private', 'features.csv');
    if (fs.existsSync(fpath)) {
      const flines = fs.readFileSync(fpath, 'utf8').trim().split('\n').slice(1);
      await pool.query('DELETE FROM features');
      for (let i = 0; i < flines.length; i += 500) {
        const chunk = flines.slice(i, i + 500).map((l) => {
          const firstComma = l.indexOf(','), secondComma = l.indexOf(',', firstComma + 1), thirdComma = l.indexOf(',', secondComma + 1);
          let payload = l.slice(thirdComma + 1);
          // pandas' to_csv quotes this field (JSON syntax always contains '"') and doubles internal quotes; undo that minimal CSV quoting
          if (payload.startsWith('"') && payload.endsWith('"')) payload = payload.slice(1, -1).replace(/""/g, '"');
          return [l.slice(0, firstComma), l.slice(firstComma + 1, secondComma), Number(l.slice(secondComma + 1, thirdComma)), payload];
        });
        const params = []; const values = chunk.map((c, j) => { params.push(c[0], c[1], c[2], c[3]); return `($${j * 4 + 1},$${j * 4 + 2},$${j * 4 + 3},$${j * 4 + 4})`; });
        await pool.query(`INSERT INTO features (series, review_date, horizon, payload) VALUES ${values.join(',')}`, params);
      }
    }
  }
```

This mirrors the existing `quantiles` loading block's chunking style. The manual comma-splitting (rather than a naive `l.split(',')`) is because the `payload` column is itself a JSON string, which legitimately contains commas *and* double-quote characters (JSON syntax always has them) — pandas' `to_csv` therefore always wraps this field in `"..."` and doubles any internal `"`, same as any RFC4180 writer would for a field containing its delimiter or quote character. Everything after the third comma is that one (possibly quoted) field; the `startsWith('"')` branch undoes pandas' quoting before the value reaches Postgres, since a `jsonb` column needs valid JSON text, not CSV-escaped JSON text. No `JSON.parse`/`JSON.stringify` round trip needed beyond that unquoting — `pg` sends the unquoted string as a parameter and Postgres casts it to `jsonb`.

- [ ] **Step 2: Test against the local stack**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator"
docker compose up -d db
docker compose run --rm -e DATABASE_URL=postgres://demo:demo@db:5432/demo server node scripts/load.js --quantiles
```

(This needs `demo_private/quantiles.csv` and `demo_private/features.csv` both present locally from earlier `uv run` steps — they will be, from this project's existing local setup plus Task 2.)

Expected: `loaded 5 scenarios and the quantile tables; data version ...` (same message as before — this block doesn't change the console output, only what's loaded). Verify features actually landed:

```bash
docker compose run --rm -e DATABASE_URL=postgres://demo:demo@db:5432/demo server node -e "
const {Pool}=require('pg'); const p=new Pool({connectionString:process.env.DATABASE_URL});
p.query('SELECT count(*) FROM features').then(r=>{console.log(r.rows[0]); p.end();});
"
```

Expected: a count matching Task 2's row count.

- [ ] **Step 3: Commit**

```bash
git add server/scripts/load.js
git commit -m "Load features.csv into the features table alongside quantiles"
```

---

### Task 4: Commit the servable model, fix the Dockerfile

**Files:**
- Create (git-tracked): `models/serving/models.lock.json`, `models/serving/v4_P10.meta.json`, `models/serving/v4_P10.ubj`
- Modify: `api/Dockerfile`

**Interfaces:**
- Produces: a Docker image for the model service with the model baked in at `/models`, matching `MODEL_DIR=/models` (already set in the Dockerfile) — no host volume mount needed, which Render doesn't support anyway.

- [ ] **Step 1: Check these files aren't already git-ignored for a reason**

```bash
git check-ignore -v models/serving/v4_P10.ubj
```

Expected: **no output** (not ignored). If it prints a matching `.gitignore` rule, stop and check why `models/` might be ignored (likely a blanket rule for the much larger raw model dumps in `models/*.json` at the top level, which should stay ignored — only `models/serving/` needs to become trackable; if needed, add a `!models/serving/` exception line to `.gitignore`).

- [ ] **Step 2: Add the files**

```bash
git add -f models/serving/models.lock.json models/serving/v4_P10.meta.json models/serving/v4_P10.ubj
git status --short
```

Expected: three new files staged, nothing else. (`-f` only needed if Step 1 found a blanket ignore rule that a `.gitignore` exception didn't fully cover — omit `-f` if `git check-ignore` printed nothing.)

- [ ] **Step 3: Update the Dockerfile to bake the model in**

Modify `api/Dockerfile` — replace the comment-only line about mounting with an actual `COPY`:

```dockerfile
# build context: the repository root. The models ARE in the image (committed at models/serving/, hash-verified via models.lock.json).
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.53.0 xgboost==3.4.1 numpy==2.2.6
COPY api/serving.py api/app.py ./
COPY models/serving /models
ENV MODEL_DIR=/models LOCK_FILE=/models/models.lock.json MODEL_NAME=v4_P10
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

(Only the first comment line and the new `COPY models/serving /models` line change; everything else is identical to the existing file.)

- [ ] **Step 4: Build and verify the image locally**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator"
docker build -f api/Dockerfile -t model-api-test .
docker run --rm -p 127.0.0.1:8000:8000 -e API_KEY=test-key model-api-test &
sleep 3
curl -s http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","model_version":"v4-..."}` (whatever version string is in `models.lock.json` — confirms the model loaded from the baked-in files, no bind mount). Stop the container afterward:

```bash
docker ps --filter ancestor=model-api-test -q | xargs -r docker stop
```

- [ ] **Step 5: Commit**

```bash
git add api/Dockerfile
git commit -m "Bake the servable model into the Docker image; commit models/serving/"
```

---

### Task 5: `/api/whatif/live` endpoint

**Files:**
- Modify: `server/src/app.js`
- Modify: `server/src/server.js`
- Test: `server/test/app.test.js`

**Interfaces:**
- Consumes: `createApp({ db, expectedDataVersion, corsOrigin, now, limits, modelApiUrl, modelApiKey })` — two new optional config fields.
- Produces: `GET /api/whatif/live?series=&date=&horizon=&alpha=&position=` — same query contract as the existing `/api/whatif`. Response body on success: `{series, date, horizon, alpha, quantile, order_up_to, position, order_quantity}` (same shape as `/api/whatif`, minus `risk`/`tail_note` — the live path proves the model round trip, the precomputed path already shows risk framing).

- [ ] **Step 1: Write the failing tests**

Add to `server/test/app.test.js`, after the existing `'what-if input validation and missing rows'` test:

```js
// ---- live inference: proxies to the model service, never leaks its key or its errors ----
const frow = (payload = { mean_28: 2.0 }) => fakeDb({
  'FROM quantiles': (p) => (p[0] === 'FOODS_3_090_CA_3_evaluation' ? [ROW] : []),
  'FROM features': (p) => (p[0] === 'FOODS_3_090_CA_3_evaluation' ? [{ payload }] : []),
});

test('what-if/live proxies to the model service and applies the same order-quantity math', async (t) => {
  t.mock.method(global, 'fetch', async (url, opts) => {
    assert.equal(url, 'https://model.example.com/quantiles');
    assert.equal(opts.headers['X-API-Key'], 'test-model-key');
    assert.deepEqual(JSON.parse(opts.body), { features: { mean_28: 2.0 } });
    return { ok: true, json: async () => ({ horizon: 10, quantiles: { q10: 0, q50: 4, q80: 8, q90: 9.5, q95: 12.01, q99: 15.1 } }) };
  });
  await serve(mk(frow(), { modelApiUrl: 'https://model.example.com', modelApiKey: 'test-model-key' }), async (b) => {
    const r = await get(b, `/api/whatif/live?${Q}&alpha=0.95&position=5`);
    assert.equal(r.status, 200);
    assert.equal(r.body.order_up_to, 13); // q95 12.01 -> ceil 13, same ceilUnits rule as the precomputed path
    assert.equal(r.body.order_quantity, 8);
    assert.equal(r.body.risk, undefined); // live path doesn't repeat the risk framing
  });
});

test('what-if/live returns 503, never the upstream body, when the model service errors or is unreachable', async (t) => {
  t.mock.method(global, 'fetch', async () => ({ ok: false, status: 500, json: async () => ({ detail: 'internal', secret: 'leak-me' }) }));
  await serve(mk(frow(), { modelApiUrl: 'https://model.example.com', modelApiKey: 'k' }), async (b) => {
    const r = await get(b, `/api/whatif/live?${Q}&alpha=0.95&position=5`);
    assert.equal(r.status, 503);
    assert.deepEqual(r.body, { error: 'model service unavailable' });
  });
  t.mock.method(global, 'fetch', async () => { throw new Error('ECONNREFUSED'); });
  await serve(mk(frow(), { modelApiUrl: 'https://model.example.com', modelApiKey: 'k' }), async (b) => {
    const r = await get(b, `/api/whatif/live?${Q}&alpha=0.95&position=5`);
    assert.equal(r.status, 503); // a network failure is also "model service unavailable", same as a non-2xx response — spec's error-handling rule covers both
    assert.deepEqual(r.body, { error: 'model service unavailable' });
  });
});

test('what-if/live is 404 when there is no stored feature row for that series/date/horizon (outside v4\'s date slice)', async (t) => {
  t.mock.method(global, 'fetch', async () => { throw new Error('must not be called'); });
  await serve(mk(frow(), { modelApiUrl: 'https://model.example.com', modelApiKey: 'k' }), async (b) => {
    const r = await get(b, '/api/whatif/live?series=OTHER_1&date=2016-05-08&horizon=10&alpha=0.9&position=1');
    assert.equal(r.status, 404);
  });
});

test('what-if/live is 503 without ever calling fetch when no model service is configured', async (t) => {
  t.mock.method(global, 'fetch', async () => { throw new Error('must not be called'); });
  await serve(mk(frow()), async (b) => { // no modelApiUrl passed
    assert.deepEqual((await get(b, `/api/whatif/live?${Q}&alpha=0.95&position=5`)).body, { error: 'model service unavailable' });
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator\server"
npm test
```

Expected: FAIL — `/api/whatif/live` doesn't exist yet (404s where the tests expect 200/503/etc), and `mk(frow(), {...})` config fields are silently ignored by the current `createApp` signature.

- [ ] **Step 3: Implement `featureRow` and the route**

In `server/src/app.js`, add after the existing `quantileRow` function (and before `app.get('/api/quantiles', ...)`):

```js
  async function featureRow(query) {
    const { series, date, horizon } = query;
    const h = Number(horizon);
    if (!/^[A-Za-z0-9_]{1,60}$/.test(String(series)) || !/^\d{4}-\d{2}-\d{2}$/.test(String(date)) || !HORIZONS.has(h)) return { status: 400 };
    const r = await db.query('SELECT payload FROM features WHERE series = $1 AND review_date = $2 AND horizon = $3', [series, date, h]);
    return r.rows.length ? { payload: r.rows[0].payload, h } : { status: 404 };
  }

  async function fetchWithTimeout(url, opts, ms) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ms);
    try { return await fetch(url, { ...opts, signal: ctl.signal }); } finally { clearTimeout(timer); }
  }
```

Then add the route itself, after the existing `/api/whatif` handler (before the `app.use((req, res) => res.status(404)...)` catch-all):

```js
  // live inference: the same order-quantity math as /api/whatif, but the quantile comes from a real model call, not a stored value.
  // Only covers the (series, date) pairs the v4 model actually served (docs/superpowers/specs/2026-09-23-live-per-series-inference-design.md).
  app.get('/api/whatif/live', async (req, res, next) => {
    try {
      const alpha = parseAlpha(req.query.alpha);
      const position = parsePosition(req.query.position);
      if (alpha === null || position === null) return res.status(400).json({ error: 'bad request' });
      const f = await featureRow(req.query);
      if (f.status) return res.status(f.status).json({ error: f.status === 400 ? 'bad request' : 'not found' });
      if (!modelApiUrl) return res.status(503).json({ error: 'model service unavailable' });
      let r;
      try {
        r = await fetchWithTimeout(`${modelApiUrl}/quantiles`, {
          method: 'POST', headers: { 'X-API-Key': modelApiKey, 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: f.payload }),
        }, 65_000);
      } catch (e) { return res.status(503).json({ error: 'model service unavailable' }); } // network failure/timeout is also "unavailable", same as a non-2xx response
      if (!r.ok) return res.status(503).json({ error: 'model service unavailable' });
      const body = await r.json();
      const level = Number(body.quantiles[`q${Math.round(alpha * 100)}`]);
      const orderUpTo = ceilUnits(level);
      return res.json({
        series: req.query.series, date: req.query.date, horizon: f.h, alpha, quantile: level,
        order_up_to: orderUpTo, position, order_quantity: Math.max(0, orderUpTo - position),
      });
    } catch (e) { return next(e); }
  });
```

And add `modelApiUrl`/`modelApiKey` to the `createApp` destructured parameters at the top of the function:

```js
function createApp({ db, expectedDataVersion, corsOrigin = '', now = () => new Date(), limits = {}, modelApiUrl = '', modelApiKey = '' } = {}) {
```

(Note: `= {}` default added to the whole destructured argument too, only if it wasn't already there — check the current signature first; the existing tests all call `createApp({...})` with an object, so this is a safe no-op if already present.)

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npm test
```

Expected: all tests pass, including the four new ones and every pre-existing test (unchanged behavior for everything else).

- [ ] **Step 5: Wire the env vars into `server.js`**

Modify `server/src/server.js` — add two more fields to the `createApp(...)` call:

```js
const app = createApp({
  db: pool, expectedDataVersion: EXPECTED_DATA_VERSION, corsOrigin: process.env.CORS_ORIGIN || '',
  modelApiUrl: process.env.MODEL_API_URL || '', modelApiKey: process.env.MODEL_API_KEY || '',
});
```

- [ ] **Step 6: Commit**

```bash
git add server/src/app.js server/src/server.js server/test/app.test.js
git commit -m "Add GET /api/whatif/live: proxies to the model service, sits alongside the precomputed path"
```

---

### Task 6: Dashboard "Verify live" button

**Files:**
- Modify: `web/app.js`
- Modify: `web/style.css`

**Interfaces:**
- Consumes: `GET ${CFG.API_BASE}/api/whatif/live?...` (Task 5).
- Produces: a button in the existing what-if result output (`renderWhatIf()`'s `out` element).

- [ ] **Step 1: Add the button and its handler**

In `web/app.js`, inside `renderWhatIf()`'s form submit handler, after the existing `out.replaceChildren(...)` success branch (the one building the order-up-to/risk-band paragraphs), append a "Verify live" button to that same result instead of replacing it entirely. Modify the `try` block's success branch:

```js
    try {
      const r = await getJSON(`${CFG.API_BASE}/api/whatif?${q}`, CFG.TIMEOUT_MS);
      const liveBtn = h('button', { class: 'go', type: 'button', onclick: async () => {
        liveBtn.disabled = true; liveOut.replaceChildren(h('p', { class: 'muted' }, 'Calling the model service live (a cold start can take about a minute)…'));
        try {
          const lq = new URLSearchParams({ series: q.get('series'), date: q.get('date'), horizon: q.get('horizon'), alpha: q.get('alpha'), position: q.get('position') });
          const lr = await getJSON(`${CFG.API_BASE}/api/whatif/live?${lq}`, 65000);
          liveOut.replaceChildren(h('p', { class: 'label' }, `Live model service: order-up-to ${lr.order_up_to} (q = ${Number(lr.quantile).toFixed(2)}), order ${lr.order_quantity} units — should match the stored value above, since it's the same model called fresh.`));
        } catch (err) { liveOut.replaceChildren(h('p', { class: 'muted' }, 'The model service did not answer (no stored feature row for this series/date, or it is waking up). Not every date has a live-servable row — only the ones the deployed model version served.')); }
        finally { liveBtn.disabled = false; }
      } }, 'Verify live');
      const liveOut = h('div', { 'aria-live': 'polite' });
      out.replaceChildren(...[h('p', {}, `Order-up-to level ceil(q${r.alpha * 100}) = ${r.order_up_to} (q = ${Number(r.quantile).toFixed(2)}); position ${r.position}; `, h('strong', {}, `order ${r.order_quantity} units`), '.'),
        h('p', {}, 'Risk band: ', h('span', { class: `band ${r.risk.band}` }, r.risk.band), r.risk.overstock ? ' (overstock flag)' : '', ' ', h('span', { class: 'label' }, r.risk.note)),
        r.risk.qualifiers.length ? h('p', { class: 'label' }, 'Qualifiers: ' + r.risk.qualifiers.join('; ')) : null, r.tail_note ? h('p', { class: 'tail' }, r.tail_note) : null,
        live ? liveBtn : null, liveOut].filter(Boolean));
    } catch (err) { out.replaceChildren(h('p', { class: 'muted' }, 'No stored quantile for that series, date and horizon, or the service is waking up (it can take about a minute). Try again.')); }
```

(This replaces the existing success-branch body of that `try` block — the catch branch is unchanged. `live ? liveBtn : null` reuses the existing `live` flag from `renderWhatIf()`'s outer scope, same one already gating the form's `disabled` attributes — no point offering "Verify live" in snapshot-only mode, since the precomputed value itself isn't live there either.)

- [ ] **Step 2: No new CSS needed**

The button reuses the existing `.go` class (already styled — same blue "Compute order" button look) and `.label`/`.muted` text classes already in `web/style.css`. Skip this file — confirmed by reading the existing stylesheet, `.go` and `.label` already exist and need no changes.

- [ ] **Step 3: Manual browser verification**

With the local Docker stack up (`docker compose up -d db server`, per this project's existing local-dev routine) and a features row loaded (Task 3):

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator\web"
python -m http.server 8080 --bind 127.0.0.1
```

Open `http://127.0.0.1:8080/index.html`, go to the "Order what-if" section, fill in a series/date that's actually in `demo_private/features.csv` (check with `uv run python -c "import pandas as pd; print(pd.read_csv('demo_private/features.csv')[['series','review_date']].head())"` for a real example), submit, then click "Verify live" — confirm it either shows a live result (if the local model-api container is also running via `docker compose --profile models up -d model-api`) or the graceful "did not answer" message (if not).

- [ ] **Step 4: Commit**

```bash
git add web/app.js
git commit -m "Add 'Verify live' button: proves the deployed model service actually answers"
```

---

### Task 7: Live-vs-precomputed agreement test (Python)

**Files:**
- Create: `tests/test_live_inference_matches_precomputed.py`

**Interfaces:**
- Consumes: `demo_private/features.csv`, `demo_private/quantiles.csv` (both local-only, git-ignored), `models/serving/v4_P10.*` (Task 4), `api/serving.py` (existing, unmodified).
- Produces: one `@pytest.mark.realdata` test — the strongest correctness check available, since it proves the live path and the precomputed path agree because they're the same model.

- [ ] **Step 1: Write the test**

```python
"""Live inference (api/serving.py, called with a stored features.csv row) must reproduce the value already
stored in quantiles.csv for the same (series, date, horizon) — they're the same v4 model, so they must agree.
Needs demo_private/features.csv and demo_private/quantiles.csv (git-ignored, local pipeline output); skipped otherwise."""
import json
from pathlib import Path

import pandas as pd
import pytest

import serving
from config import ROOT

FEATURES = ROOT / "demo_private" / "features.csv"
QUANTILES = ROOT / "demo_private" / "quantiles.csv"


@pytest.mark.realdata
@pytest.mark.skipif(not (FEATURES.exists() and QUANTILES.exists()), reason="demo_private/*.csv not built locally")
def test_live_model_output_matches_the_stored_quantile_for_the_same_row():
    feats = pd.read_csv(FEATURES)
    quant = pd.read_csv(QUANTILES)
    merged = feats.merge(quant, on=["series", "review_date", "horizon"], how="inner")
    assert len(merged) > 0, "no overlap between features.csv and quantiles.csv rows — check versions.use_dates(4) filtering"
    booster, meta = serving.load(ROOT / "models" / "serving", "v4_P10")
    sample = merged.sample(n=min(20, len(merged)), random_state=0)
    for _, row in sample.iterrows():
        got = serving.predict_quantiles(booster, meta, [json.loads(row["payload"])])[0]
        want = [row.q10, row.q50, row.q80, row.q90, row.q95, row.q99]
        assert list(got) == pytest.approx(want, abs=1e-3), (row["series"], row["review_date"])
```

- [ ] **Step 2: Run it**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator"
uv run pytest tests/test_live_inference_matches_precomputed.py -m realdata -v
```

Expected: `PASSED` if `demo_private/features.csv` and `demo_private/quantiles.csv` both exist locally (they will, from Task 2 and this project's existing setup) — confirms the live model call reproduces the already-shipped numbers exactly, for the rows where the two datasets actually overlap. If it's `SKIPPED`, that's also a valid outcome on a machine without the real data pipeline run (e.g., CI) — not a failure.

- [ ] **Step 3: Confirm the fast suite still skips this by default**

```bash
uv run pytest tests/test_live_inference_matches_precomputed.py -v
```

Expected: `SKIPPED` (or deselected) without the `-m realdata` flag — same behavior as every other real-data test in this repo (`pyproject.toml`'s `addopts = "-m 'not realdata'"`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_live_inference_matches_precomputed.py
git commit -m "Add realdata test: live model output matches the precomputed quantiles for the same row"
```

---

### Task 8: Deploy the model service to Render

This task is operational (Render dashboard clicks + verification curls), not a code change — no automated test applies. Whoever executes this plan does it by hand or guides the repository owner through it, same pattern already used earlier in this project's Supabase/Render setup.

- [ ] **Step 1: Push everything so far**

```bash
cd "D:\Github Projects\Probabilistic Demand Forecasting Simulator"
git push origin main
```

- [ ] **Step 2: Add a second service to `render.yaml`**

Modify `render.yaml` — add alongside the existing `demand-forecast-demo-api` service:

```yaml
  - type: web
    name: demand-forecast-model-api
    runtime: docker
    plan: free
    dockerfilePath: ./api/Dockerfile
    dockerContext: .
    healthCheckPath: /health
    envVars:
      - key: API_KEY
        sync: false
```

Commit:
```bash
git add render.yaml
git commit -m "Add the model service to the Render blueprint"
git push origin main
```

- [ ] **Step 3: Deploy via the Render dashboard**

Render dashboard → the existing Blueprint (or **New** → **Blueprint** again if it needs re-sync) → it should now show the new `demand-forecast-model-api` service alongside the existing one → apply. Once created, **Environment** tab → set `API_KEY` to a freshly generated random value (e.g. `openssl rand -hex 32` run locally — never reuse a memorable password here).

- [ ] **Step 4: Verify it's live**

```bash
curl -s https://demand-forecast-model-api.onrender.com/health
```

Expected: `{"status":"ok","model_version":"..."}` (matching what Task 4's local Docker test showed).

- [ ] **Step 5: Set the same key on the Express service**

Render dashboard → `demand-forecast-demo-api` (or the Vercel project once Task 9 is done, whichever is currently live) → environment variables → `MODEL_API_URL=https://demand-forecast-model-api.onrender.com`, `MODEL_API_KEY=<the same key from Step 3>`.

---

### Task 9: Deploy Express + static site to Vercel

- [ ] **Step 1: Wrap the Express app as a Vercel function**

Create `server/api/index.js` (Vercel's own convention — distinct from the repository's top-level `api/` FastAPI folder):

```js
const { Pool } = require('pg');
const { createApp } = require('../src/app');
const { EXPECTED_DATA_VERSION } = require('../src/dataVersion');

let app;
function getApp() {
  if (!app) {
    const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 1, ssl: { rejectUnauthorized: false }, connectionTimeoutMillis: 5000 });
    app = createApp({
      db: pool, expectedDataVersion: EXPECTED_DATA_VERSION, corsOrigin: process.env.CORS_ORIGIN || '',
      modelApiUrl: process.env.MODEL_API_URL || '', modelApiKey: process.env.MODEL_API_KEY || '',
    });
  }
  return app;
}

module.exports = (req, res) => getApp()(req, res);
```

`max: 1` (versus `server.js`'s `max: 4`) because this runs as a serverless function — each cold start gets its own instance, so a pool sized for a long-lived container would hold idle connections needlessly against Supabase's transaction pooler.

Create `server/vercel.json`:
```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/api" }] }
```

- [ ] **Step 2: Create the Vercel project for the API**

Use the Vercel MCP tools available in this session: list the account's teams/projects first (per the Vercel MCP server's own instructions already loaded in this session — discover team/project scope before creating), then create a project rooted at `server/`, connected to the same GitHub repo. Set env vars: `DATABASE_URL` (the Supabase pooler string already in use), `CORS_ORIGIN`, `MODEL_API_URL`, `MODEL_API_KEY` (from Task 8).

- [ ] **Step 3: Create the Vercel project for the static site**

Same approach, rooted at `web/`, no build command (static files as-is).

- [ ] **Step 4: Update `web/config.js`**

```js
window.DEMO_CONFIG = { API_BASE: '<the Express Vercel project URL from Step 2>', TIMEOUT_MS: 8000, WAKING_AFTER_MS: 3000 };
```

Commit and push:
```bash
git add server/api/index.js server/vercel.json web/config.js
git commit -m "Deploy Express API and static site to Vercel"
git push origin main
```

- [ ] **Step 5: End-to-end verification**

```bash
curl -s https://<api-project>.vercel.app/health
curl -s https://<api-project>.vercel.app/health/deep
```

Expected: both `200 {"status":"ok",...}`. Then open the static site's URL in a browser, run through the what-if flow, click "Verify live" for a series/date that's in `demo_private/features.csv`, confirm the live result appears and roughly matches the precomputed one (per Task 7's test, they should agree closely for rows in the v4 slice).

---

## Self-Review Notes

- **Spec coverage:** A (Task 2), B/schema+loader (Tasks 1, 3), C/model service (Tasks 4, 8), D/Express (Task 5), E/static site (Task 9), F/UI (Task 6), G/tests (Tasks 5's inline tests + Task 7) — every spec section has a task.
- **Type consistency checked:** `featureRow` returns `{ payload, h }` or `{ status }`, matching `quantileRow`'s existing `{ row, h }` / `{ status }` shape (renamed `row`→`payload` since it's a single jsonb value, not a row of named columns) — used consistently in Task 5's route handler and tests.
- **No placeholders:** every step has literal code or literal commands; deployment tasks (8, 9) are runbook-style by nature (dashboard actions can't be unit tested) but every command in them is concrete and copy-pasteable.
