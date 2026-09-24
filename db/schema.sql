-- Precomputed results served by the demo API (server/). Loaded by server/scripts/load.js.
CREATE TABLE IF NOT EXISTS data_version (version text PRIMARY KEY, built_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS scenarios (id text PRIMARY KEY, position int NOT NULL, title text NOT NULL, payload jsonb NOT NULL);
-- per-series quantile tables: model outputs derived from M5 sales, loaded only into a LOCAL database until the M5 data-use terms are confirmed
CREATE TABLE IF NOT EXISTS quantiles (
  series text NOT NULL, review_date date NOT NULL, horizon int NOT NULL,
  q10 real NOT NULL, q50 real NOT NULL, q80 real NOT NULL, q90 real NOT NULL, q95 real NOT NULL, q99 real NOT NULL, zero_run boolean NOT NULL DEFAULT false,
  PRIMARY KEY (series, review_date, horizon)
);
-- per-series model input features for live inference: covers every test-window review date, each tagged with the
-- model version that serves it (versions.use_dates), same publication gate as `quantiles`
CREATE TABLE IF NOT EXISTS features (
  series text NOT NULL, review_date date NOT NULL, horizon int NOT NULL,
  payload jsonb NOT NULL,
  PRIMARY KEY (series, review_date, horizon)
);
-- migration: existing rows were all v4_P10 (the only version ever loaded before); the loader overwrites them anyway
ALTER TABLE features ADD COLUMN IF NOT EXISTS model text NOT NULL DEFAULT 'v4_P10';
