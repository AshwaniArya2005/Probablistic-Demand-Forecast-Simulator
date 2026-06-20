const { Pool } = require('pg');
const { createApp } = require('./app');
const { EXPECTED_DATA_VERSION } = require('./dataVersion');

if (!process.env.DATABASE_URL) {
  console.error('DATABASE_URL is not set');
  process.exit(1);
}
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 4, ssl: process.env.DATABASE_SSL === 'off' ? false : { rejectUnauthorized: false }, connectionTimeoutMillis: 5000 });
// an idle client can error when the database restarts; without a handler the process would crash instead of reporting 503 on /health/deep
pool.on('error', (e) => console.error('database pool error:', e.code || 'unknown'));
const app = createApp({ db: pool, expectedDataVersion: EXPECTED_DATA_VERSION, corsOrigin: process.env.CORS_ORIGIN || '' });
const port = Number(process.env.PORT || 3000);
app.listen(port, () => console.log(`listening on ${port}`));
