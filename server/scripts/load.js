// Loads db/schema.sql, web/snapshot.json (scenarios) and, optionally, demo_private/quantiles.csv into Postgres.
//   DATABASE_URL=... node scripts/load.js                 scenarios only (aggregates: safe to publish)
//   DATABASE_URL=... node scripts/load.js --quantiles     also the per-series quantile tables; refused for a non-local database
//                                                          unless ALLOW_PUBLIC_QUANTILES=confirmed-kaggle-rules (see docs/design.md, Phase 11)
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const { EXPECTED_DATA_VERSION } = require('../src/dataVersion');

const root = path.join(__dirname, '..', '..');
const url = process.env.DATABASE_URL;
if (!url) { console.error('DATABASE_URL is not set'); process.exit(1); }
const local = /@(localhost|127\.0\.0\.1|db)(:|\/)/.test(url);
const withQuantiles = process.argv.includes('--quantiles');
if (withQuantiles && !local && process.env.ALLOW_PUBLIC_QUANTILES !== 'confirmed-kaggle-rules') {
  console.error('refusing to load per-series quantile tables into a non-local database: the M5 data-use terms have not been confirmed');
  process.exit(2);
}

(async () => {
  const pool = new Pool({ connectionString: url, ssl: local ? false : { rejectUnauthorized: false } });
  const snap = JSON.parse(fs.readFileSync(path.join(root, 'web', 'snapshot.json'), 'utf8'));
  if (snap.data_version !== EXPECTED_DATA_VERSION) throw new Error(`snapshot version ${snap.data_version} differs from the server's ${EXPECTED_DATA_VERSION}`);
  await pool.query(fs.readFileSync(path.join(root, 'db', 'schema.sql'), 'utf8'));
  await pool.query('DELETE FROM scenarios');
  for (const [i, s] of snap.scenarios.entries()) await pool.query('INSERT INTO scenarios (id, position, title, payload) VALUES ($1, $2, $3, $4)', [s.id, i, s.title, JSON.stringify(s)]);
  if (withQuantiles) {
    const lines = fs.readFileSync(path.join(root, 'demo_private', 'quantiles.csv'), 'utf8').trim().split('\n').slice(1);
    await pool.query('DELETE FROM quantiles');
    for (let i = 0; i < lines.length; i += 500) {
      const chunk = lines.slice(i, i + 500).map((l) => l.split(','));
      const params = []; const values = chunk.map((c, j) => { params.push(c[0], c[1], Number(c[2]), ...c.slice(3, 9).map(Number), c[9] === 'True'); return `(${Array.from({ length: 10 }, (_, k) => `$${j * 10 + k + 1}`).join(',')})`; });
      await pool.query(`INSERT INTO quantiles (series, review_date, horizon, q10, q50, q80, q90, q95, q99, zero_run) VALUES ${values.join(',')}`, params);
    }
  }
  if (withQuantiles) {
    const fpath = path.join(root, 'demo_private', 'features.csv');
    if (fs.existsSync(fpath)) {
      // split(/\r?\n/) not split('\n'): pandas' to_csv writes CRLF on Windows, and a stray '\r'
      // left on the payload field would break both the unquoting check below and the resulting JSON
      const flines = fs.readFileSync(fpath, 'utf8').trim().split(/\r?\n/).slice(1);
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
  // the data version is written last: /health/deep only reports ok when every table above finished loading
  await pool.query('DELETE FROM data_version');
  await pool.query('INSERT INTO data_version (version) VALUES ($1)', [EXPECTED_DATA_VERSION]);
  console.log(`loaded ${snap.scenarios.length} scenarios${withQuantiles ? ' and the quantile tables' : ''}; data version ${EXPECTED_DATA_VERSION}`);
  await pool.end();
})().catch((e) => { console.error('load failed:', e.message); process.exit(1); });
