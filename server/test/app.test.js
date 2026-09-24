const test = require('node:test');
const assert = require('node:assert/strict');
const { createApp, ceilUnits, riskBand } = require('../src/app');

const SECRET = 'postgres://user:hunter2@db.example.com:5432/prod';
const EXPECTED = 'v1-test';

// a fake database: `handlers` maps a fragment of the SQL text to a function returning rows (or throwing)
function fakeDb(handlers = {}) {
  const base = { 'SELECT 1': () => [{ '?column?': 1 }], 'FROM data_version': () => [{ version: EXPECTED }], ...handlers };
  return {
    async query(sql, params) {
      const k = Object.keys(base).find((f) => sql.includes(f));
      if (!k) throw new Error('unexpected sql ' + sql);
      return { rows: base[k](params) };
    },
  };
}

async function serve(app, fn) {
  const server = await new Promise((resolve) => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  try { return await fn(base); } finally { await new Promise((resolve) => server.close(resolve)); }
}
// uses node:http rather than global fetch, so tests that mock global.fetch (the app's outbound model-service call) don't also intercept this
const http = require('node:http');
const get = (base, path) => new Promise((resolve, reject) => {
  http.get(base + path, (res) => {
    let data = '';
    res.on('data', (c) => { data += c; });
    res.on('end', () => resolve({
      status: res.statusCode,
      body: JSON.parse(data),
      headers: { get: (k) => res.headers[k.toLowerCase()] ?? null },
    }));
  }).on('error', reject);
});
const mk = (db, extra = {}) => createApp({ db, expectedDataVersion: EXPECTED, ...extra });

test('GET /health returns 200 with the version and calls no dependency', async () => {
  const db = { async query() { throw new Error('must not be called'); } };
  await serve(mk(db), async (b) => {
    const r = await get(b, '/health');
    assert.equal(r.status, 200);
    assert.deepEqual(Object.keys(r.body).sort(), ['status', 'version']);
    assert.equal(r.body.status, 'ok');
  });
});

test('GET /health/deep is 200 when the database answers and the data version matches, and /api/status records it', async () => {
  await serve(mk(fakeDb(), { now: () => new Date('2026-09-22T10:00:00Z') }), async (b) => {
    assert.deepEqual((await get(b, '/api/status')).body, { last_checked: null, ok: null });
    const r = await get(b, '/health/deep');
    assert.equal(r.status, 200);
    assert.deepEqual(Object.keys(r.body).sort(), ['status', 'version']);
    assert.deepEqual((await get(b, '/api/status')).body, { last_checked: '2026-09-22T10:00:00.000Z', ok: true });
  });
});

test('GET /health/deep is 503 with a short reason when the database is down, and the error detail never leaks', async () => {
  const db = fakeDb({ 'SELECT 1': () => { throw new Error(`connect ECONNREFUSED ${SECRET}`); } });
  await serve(mk(db), async (b) => {
    const r = await get(b, '/health/deep');
    assert.equal(r.status, 503);
    assert.deepEqual(r.body, { status: 'unavailable', reason: 'database' });
    assert.ok(!JSON.stringify(r.body).includes('hunter2') && !JSON.stringify(r.body).includes('postgres'));
    assert.equal((await get(b, '/api/status')).body.ok, false);
  });
});

test('GET /health/deep is 503 "data version" when the precomputed data is not the expected version, or missing', async () => {
  await serve(mk(fakeDb({ 'FROM data_version': () => [{ version: 'v0-old' }] })), async (b) => {
    assert.deepEqual((await get(b, '/health/deep')).body, { status: 'unavailable', reason: 'data version' });
  });
  await serve(mk(fakeDb({ 'FROM data_version': () => [] })), async (b) => {
    const r = await get(b, '/health/deep');
    assert.equal(r.status, 503);
    assert.equal(r.body.reason, 'data version');
  });
  await serve(mk(fakeDb({ 'FROM data_version': () => { throw new Error('relation "data_version" does not exist'); } })), async (b) => {
    assert.equal((await get(b, '/health/deep')).body.reason, 'database');
  });
});

test('no health response leaks secrets, hashes, paths or configuration', async () => {
  process.env.DATABASE_URL = SECRET;
  await serve(mk(fakeDb()), async (b) => {
    for (const p of ['/health', '/health/deep', '/api/status']) {
      const text = JSON.stringify((await get(b, p)).body);
      for (const banned of ['hunter2', 'postgres', 'sha256', 'models.lock', 'password', 'DATABASE']) assert.ok(!text.includes(banned), `${p} contains ${banned}`);
      assert.ok(!/[0-9a-f]{32,}/.test(text), `${p} looks like it contains a hash`);
      assert.ok(!/[\\/]/.test(text.replace(/"/g, '').replace(/T\d\d:\d\d:\d\d/g, '')) || p === '/api/status', `${p} contains a path separator`);
    }
  });
  delete process.env.DATABASE_URL;
});

test('errors elsewhere are generic and carry no detail', async () => {
  const db = fakeDb({ 'FROM scenarios': () => { throw new Error(`boom ${SECRET}`); } });
  await serve(mk(db), async (b) => {
    const r = await get(b, '/api/scenarios');
    assert.equal(r.status, 500);
    assert.deepEqual(r.body, { error: 'internal error' });
    assert.equal((await get(b, '/nope')).status, 404);
    assert.equal(r.headers.get('x-powered-by'), null);
  });
});

test('rate limiting: the general limit and the stricter /health/deep limit return 429', async () => {
  await serve(mk(fakeDb(), { limits: { general: 3, deep: 2 } }), async (b) => {
    const deep = [];
    for (let i = 0; i < 3; i += 1) deep.push((await get(b, '/health/deep')).status);
    assert.deepEqual(deep, [200, 200, 429]);
    const plain = [];
    for (let i = 0; i < 4; i += 1) plain.push((await get(b, '/health')).status);
    assert.deepEqual(plain, [200, 200, 200, 429]);
  });
});

// ---- the what-if and the risk band, by hand ----
const ROW = { q10: 0, q50: 4.2, q80: 8, q90: 9.5, q95: 12.01, q99: 15.1, zero_run: false };
const qdb = (row = ROW) => fakeDb({ 'FROM quantiles': (p) => (p[0] === 'FOODS_3_090_CA_3_evaluation' ? [row] : []) });
const Q = 'series=FOODS_3_090_CA_3_evaluation&date=2016-05-08&horizon=10';

test('ceilUnits ignores arithmetic noise but not a real fraction', () => {
  assert.equal(ceilUnits(12), 12);
  assert.equal(ceilUnits(12.000000000000002), 12);
  assert.equal(ceilUnits(12.01), 13);
  assert.equal(ceilUnits(0), 0);
  assert.equal(ceilUnits(0.001), 1);
  assert.equal(ceilUnits(-0.3), 0);
});

test('what-if order quantity = max(0, ceil(q) - position), by hand', async () => {
  await serve(mk(qdb()), async (b) => {
    let r = await get(b, `/api/whatif?${Q}&alpha=0.95&position=5`); // q95 = 12.01 -> 13; 13 - 5 = 8
    assert.equal(r.status, 200);
    assert.equal(r.body.order_up_to, 13);
    assert.equal(r.body.order_quantity, 8);
    r = await get(b, `/api/whatif?${Q}&alpha=0.95&position=20`); // already above the level: order nothing
    assert.equal(r.body.order_quantity, 0);
    r = await get(b, `/api/whatif?${Q}&alpha=0.9&position=0`); // q90 = 9.5 -> 10
    assert.equal(r.body.order_quantity, 10);
    assert.equal(r.body.tail_note, undefined);
    r = await get(b, `/api/whatif?${Q}&alpha=0.99&position=0`); // q99 = 15.1 -> 16, marked as the costly tail
    assert.equal(r.body.order_quantity, 16);
    assert.match(r.body.tail_note, /costly tail/);
  });
});

test('risk band boundaries (Q50 = 5, Q90 = 10, Q99 = 16) and the overstock flag', () => {
  const at = (p) => riskBand(p, 4.2, 9.5, 15.1, false);
  assert.deepEqual([4, 5, 9, 10, 16, 17].map((p) => at(p).band), ['HIGH', 'MEDIUM', 'MEDIUM', 'LOW', 'LOW', 'LOW']);
  assert.deepEqual([4, 5, 9, 10, 16, 17].map((p) => at(p).overstock), [false, false, false, false, false, true]);
  assert.match(at(4).note, /not a probability/);
});

test('both sides of a ceiled tie: an exact integer quantile is its own cut point, just above rounds up', () => {
  assert.equal(riskBand(4, 5.0, 9, 12, false).band, 'HIGH');
  assert.equal(riskBand(5, 5.0, 9, 12, false).band, 'MEDIUM');
  assert.equal(riskBand(5, 5.01, 9, 12, false).band, 'HIGH');
  assert.equal(riskBand(6, 5.01, 9, 12, false).band, 'MEDIUM');
  assert.equal(riskBand(5, 5.0000000004, 9, 12, false).band, 'MEDIUM');
});

test('the two qualifiers of the label investigation are attached', () => {
  assert.deepEqual(riskBand(0, 0.2, 3, 6, false).qualifiers, ['median below one unit']);
  assert.deepEqual(riskBand(0, 3, 4, 6, true).qualifiers, ['long zero run: sales history may be censored']);
  assert.deepEqual(riskBand(9, 0.2, 3, 6, false).qualifiers, []);
});

test('what-if input validation and missing rows', async () => {
  await serve(mk(qdb()), async (b) => {
    const bad = [
      `${Q}&alpha=0.5&position=1`, `${Q}&alpha=0.9&position=-1`, `${Q}&alpha=0.9&position=abc`, `${Q}&alpha=0.9`,
      'series=x&date=2016-05-08&horizon=7&alpha=0.9&position=1', 'series=a%27%3BDROP&date=2016-05-08&horizon=10&alpha=0.9&position=1',
      'series=FOODS_3_090_CA_3_evaluation&date=yesterday&horizon=10&alpha=0.9&position=1',
    ];
    for (const q of bad) assert.equal((await get(b, `/api/whatif?${q}`)).status, 400, q);
    assert.equal((await get(b, '/api/whatif?series=OTHER_1&date=2016-05-08&horizon=10&alpha=0.9&position=1')).status, 404);
    const r = await get(b, `/api/quantiles?${Q}`);
    assert.equal(r.status, 200);
    assert.equal(r.body.quantiles.q90, 9.5);
  });
});

// ---- live inference: proxies to the model service, never leaks its key or its errors ----
const frow = (payload = { mean_28: 2.0 }, model = 'v4_P10') => fakeDb({
  'FROM quantiles': (p) => (p[0] === 'FOODS_3_090_CA_3_evaluation' ? [ROW] : []),
  'FROM features': (p) => (p[0] === 'FOODS_3_090_CA_3_evaluation' ? [{ payload, model }] : []),
});

test('what-if/live proxies to the model service and applies the same order-quantity math', async (t) => {
  t.mock.method(global, 'fetch', async (url, opts) => {
    assert.equal(url, 'https://model.example.com/quantiles');
    assert.equal(opts.headers['X-API-Key'], 'test-model-key');
    assert.deepEqual(JSON.parse(opts.body), { features: { mean_28: 2.0 }, model: 'v4_P10' });
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

test('what-if/live is 404 when there is no stored feature row for that series/date/horizon', async (t) => {
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

test('scenarios are read from the database and only listed ids are returned', async () => {
  const db = fakeDb({ 'ORDER BY position': () => [{ id: 'primary', title: 'Base case' }], 'WHERE id = $1': (p) => (['primary', 'L7'].includes(p[0]) ? [{ payload: { title: p[0] === 'L7' ? 'Lead time 7' : 'Base case' } }] : []) });
  await serve(mk(db), async (b) => {
    assert.deepEqual((await get(b, '/api/scenarios')).body, [{ id: 'primary', title: 'Base case' }]);
    assert.equal((await get(b, '/api/scenarios/primary')).body.title, 'Base case');
    assert.equal((await get(b, '/api/scenarios/L7')).body.title, 'Lead time 7'); // ids may contain capitals (the real scenario id is L7)
    assert.equal((await get(b, '/api/scenarios/unknown')).status, 404);
    assert.equal((await get(b, '/api/scenarios/BAD%20ID')).status, 400);
  });
});

test('GET /api/series lists distinct series ids for the what-if form', async () => {
  const db = fakeDb({ 'DISTINCT series': () => [{ series: 'FOODS_3_090_CA_3_evaluation' }, { series: 'HOBBIES_1_001_CA_1_evaluation' }] });
  await serve(mk(db), async (b) => {
    assert.deepEqual((await get(b, '/api/series')).body, ['FOODS_3_090_CA_3_evaluation', 'HOBBIES_1_001_CA_1_evaluation']);
  });
});
