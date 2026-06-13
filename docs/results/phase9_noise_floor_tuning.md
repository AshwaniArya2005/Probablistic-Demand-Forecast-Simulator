# Forecast error against a Poisson noise floor (tuning folds, report-only)

Run 2026-09-21. `WAPE_forecast` is the normalised-target XGBoost mean forecast used by the B2 and B3a policies (Phase 9 rows: 12 Sunday origins x 4 folds x 300 series per horizon);
`WAPE_poisson_floor` is what a forecast equal to the truth would score if demand were Poisson around it. Real demand is overdispersed (promotions, SNAP days, stock-outs), so the floor cannot be reached.

| P   | segment   |     n |   mean_demand |   WAPE_forecast |   WAPE_MA28 |   WAPE_poisson_floor |
|:----|:----------|------:|--------------:|----------------:|------------:|---------------------:|
| 7   | pooled    | 14400 |         9.094 |           0.412 |       0.412 |                0.227 |
| 7   | high      |  4800 |        19.681 |           0.363 |       0.367 |                0.166 |
| 7   | low       |  4752 |         1.966 |           0.714 |       0.685 |                0.558 |
| 7   | mid       |  4848 |         5.598 |           0.478 |       0.475 |                0.324 |
| 10  | pooled    | 14400 |        12.599 |           0.391 |       0.386 |                0.195 |
| 10  | high      |  4800 |        27.313 |           0.353 |       0.349 |                0.143 |
| 10  | low       |  4752 |         2.706 |           0.651 |       0.626 |                0.484 |
| 10  | mid       |  4848 |         7.728 |           0.437 |       0.436 |                0.279 |
| 14  | pooled    | 14400 |        18.185 |           0.366 |       0.357 |                0.162 |
| 14  | high      |  4800 |        39.33  |           0.336 |       0.326 |                0.119 |
| 14  | low       |  4752 |         3.954 |           0.575 |       0.554 |                0.397 |
| 14  | mid       |  4848 |        11.197 |           0.399 |       0.393 |                0.23  |
| all | pooled    | 43200 |        13.292 |           0.385 |       0.379 |                0.187 |
| all | high      | 14400 |        28.775 |           0.348 |       0.343 |                0.137 |
| all | low       | 14256 |         2.875 |           0.631 |       0.606 |                0.461 |
| all | mid       | 14544 |         8.174 |           0.429 |       0.426 |                0.267 |
