// Express API of the demo: serves PRECOMPUTED results from Postgres (no live model). See docs/design.md, Phase 11 decisions.
const express = require('express');
const rateLimit = require('express-rate-limit');
const { version: VERSION } = require('../package.json');

const ALPHA_COL = { 0.8: 'q80', 0.9: 'q90', 0.95: 'q95', 0.99: 'q99' };
const HORIZONS = new Set([10, 14]);
// integer ceiling that ignores arithmetic noise, same rule as ml/policy.py ceil_units (round to 1e-9 first)
const ceilUnits = (x) => Math.ceil(Math.round(Math.max(x, 0) * 1e9) / 1e9);

// stockout-risk band from the quantile grid, same cut points as ml/policy.py (a risk band, not a probability)
function riskBand(position, q50, q90, q99, zeroRun) {
  const [Q50, Q90, Q99] = [q50, q90, q99].map(ceilUnits);
  const band = position < Q50 ? 'HIGH' : position < Q90 ? 'MEDIUM' : 'LOW';
  const qualifiers = [];
  if (band === 'HIGH' && q50 <= 1) qualifiers.push('median below one unit');
  if (zeroRun) qualifiers.push('long zero run: sales history may be censored');
  return { band, overstock: position > Q99, qualifiers, note: 'a risk band relative to the model distribution, not a probability of stockout' };
}

const parseAlpha = (s) => { const a = Number(s); return Object.hasOwn(ALPHA_COL, a) ? a : null; };
const parsePosition = (s) => (/^\d{1,9}$/.test(String(s)) ? Number(s) : null);

function createApp({ db, expectedDataVersion, corsOrigin = '', now = () => new Date(), limits = {}, modelApiUrl = '', modelApiKey = '' } = {}) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', 1);
  const state = { last: null };
  const general = rateLimit({ windowMs: 60_000, limit: limits.general ?? 120, standardHeaders: true, legacyHeaders: false, message: { error: 'rate limit exceeded' } });
  const deep = rateLimit({ windowMs: 60_000, limit: limits.deep ?? 30, standardHeaders: true, legacyHeaders: false, message: { error: 'rate limit exceeded' } });

  app.use((req, res, next) => {
    if (corsOrigin) res.set('Access-Control-Allow-Origin', corsOrigin);
    res.set('Cache-Control', 'no-store');
    next();
  });

  // cheap liveness: no dependency calls
  app.get('/health', general, (req, res) => res.json({ status: 'ok', version: VERSION }));

  // availability: database reachable and the precomputed data is the version this code expects. Short reason only, never details.
  app.get('/health/deep', deep, async (req, res) => {
    let reason = null;
    try {
      await db.query('SELECT 1');
      const r = await db.query('SELECT version FROM data_version LIMIT 1');
      if (!r.rows.length || r.rows[0].version !== expectedDataVersion) reason = 'data version';
    } catch (e) {
      reason = 'database';
    }
    state.last = { at: now().toISOString(), ok: reason === null };
    if (reason) return res.status(503).json({ status: 'unavailable', reason });
    return res.json({ status: 'ok', version: VERSION });
  });

  // "last checked" line of the dashboard footer: the time and outcome of the last deep check
  app.get('/api/status', general, (req, res) => res.json({ last_checked: state.last ? state.last.at : null, ok: state.last ? state.last.ok : null }));

  app.use('/api', general);

  app.get('/api/scenarios', async (req, res, next) => {
    try { res.json((await db.query('SELECT id, title FROM scenarios ORDER BY position')).rows); } catch (e) { next(e); }
  });
  app.get('/api/scenarios/:id', async (req, res, next) => {
    try {
      if (!/^[A-Za-z0-9_]{1,40}$/.test(req.params.id)) return res.status(400).json({ error: 'bad request' });
      const r = await db.query('SELECT payload FROM scenarios WHERE id = $1', [req.params.id]);
      if (!r.rows.length) return res.status(404).json({ error: 'not found' });
      return res.json(r.rows[0].payload);
    } catch (e) { return next(e); }
  });

  // the fixed set of series ids that have a stored quantile row (for the what-if form's guard rail, not paginated: ~300 rows)
  app.get('/api/series', async (req, res, next) => {
    try { res.json((await db.query('SELECT DISTINCT series FROM quantiles ORDER BY series')).rows.map((r) => r.series)); } catch (e) { return next(e); }
  });

  async function quantileRow(query) {
    const { series, date, horizon } = query;
    const h = Number(horizon);
    if (!/^[A-Za-z0-9_]{1,60}$/.test(String(series)) || !/^\d{4}-\d{2}-\d{2}$/.test(String(date)) || !HORIZONS.has(h)) return { status: 400 };
    const r = await db.query('SELECT q10, q50, q80, q90, q95, q99, zero_run FROM quantiles WHERE series = $1 AND review_date = $2 AND horizon = $3', [series, date, h]);
    return r.rows.length ? { row: r.rows[0], h } : { status: 404 };
  }

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

  app.get('/api/quantiles', async (req, res, next) => {
    try {
      const q = await quantileRow(req.query);
      if (q.status) return res.status(q.status).json({ error: q.status === 400 ? 'bad request' : 'not found' });
      const { zero_run, ...quantiles } = q.row;
      return res.json({ series: req.query.series, date: req.query.date, horizon: q.h, quantiles, zero_run });
    } catch (e) { return next(e); }
  });

  // order-quantity what-if: max(0, ceil(q_alpha) - inventory position), computed here from the stored quantile
  app.get('/api/whatif', async (req, res, next) => {
    try {
      const alpha = parseAlpha(req.query.alpha);
      const position = parsePosition(req.query.position);
      if (alpha === null || position === null) return res.status(400).json({ error: 'bad request' });
      const q = await quantileRow(req.query);
      if (q.status) return res.status(q.status).json({ error: q.status === 400 ? 'bad request' : 'not found' });
      const level = Number(q.row[ALPHA_COL[alpha]]);
      const orderUpTo = ceilUnits(level);
      return res.json({
        series: req.query.series, date: req.query.date, horizon: q.h, alpha, quantile: level, order_up_to: orderUpTo, position,
        order_quantity: Math.max(0, orderUpTo - position),
        risk: riskBand(position, Number(q.row.q50), Number(q.row.q90), Number(q.row.q99), q.row.zero_run),
        tail_note: alpha === 0.99 ? 'the 0.99 level is the costly tail: it buys little extra service for a lot of extra inventory' : undefined,
      });
    } catch (e) { return next(e); }
  });

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

  app.use((req, res) => res.status(404).json({ error: 'not found' }));
  // never echo error details (they can carry connection strings or SQL)
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => res.status(500).json({ error: 'internal error' }));
  return app;
}

module.exports = { createApp, ceilUnits, riskBand, VERSION };
