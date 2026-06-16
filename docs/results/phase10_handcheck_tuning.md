# Hand recomputation of one series (Phase 10 check, tuning)

Run 2026-09-21. One mid-velocity series (chosen by a fixed rule: the first mid-velocity series, alphabetically, with at least one lost unit in the six counted weeks under the quantile policy at alpha 0.90;
id withheld), ten review weeks (four warm-up, six counted), tuning data. Every quantity below was recomputed with plain-Python integers and lists that share no code with `ml/simulator.py` and compared with the engine:
the 28-day open-day means from raw sales (closed days excluded, as in docs/features.md), the order-up-to levels of the naive rule (c = 2) and of the quantile policy (alpha 0.90), the full day-by-day stock path, the orders, and each counted cycle's demand, lost units,
stock-days and stockout flag. The day-by-day table itself contains one series' daily sales and is kept out of the repository.

**Result: all 22 checks agree.**

| check | agrees |
|---|---|
| mu28 equals the mean of the open sales days among the 28 ending on each review date (closed days excluded, docs/features.md; float32 storage, relative tolerance 1e-6) | yes |
| naive levels ceil(2 x mu28 x P) match the engine's input | yes |
| quantile levels ceil(q90) match the engine's input | yes |
| panel sales equal the demand matrix used by the engine | yes |
| quantile alpha 0.90: end-of-day on-hand path (73 days) equals the engine's | yes |
| quantile alpha 0.90: lost units per day equal the engine's | yes |
| quantile alpha 0.90: orders placed at each review equal the engine's | yes |
| quantile alpha 0.90: counted cycle 1 (review 2015-05-31): demanded, lost, stock-days, stockout flag | yes |
| quantile alpha 0.90: counted cycle 2 (review 2015-06-07): demanded, lost, stock-days, stockout flag | yes |
| quantile alpha 0.90: counted cycle 3 (review 2015-06-14): demanded, lost, stock-days, stockout flag | yes |
| quantile alpha 0.90: counted cycle 4 (review 2015-06-21): demanded, lost, stock-days, stockout flag | yes |
| quantile alpha 0.90: counted cycle 5 (review 2015-06-28): demanded, lost, stock-days, stockout flag | yes |
| quantile alpha 0.90: counted cycle 6 (review 2015-07-05): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: end-of-day on-hand path (73 days) equals the engine's | yes |
| naive c = 2: lost units per day equal the engine's | yes |
| naive c = 2: orders placed at each review equal the engine's | yes |
| naive c = 2: counted cycle 1 (review 2015-05-31): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: counted cycle 2 (review 2015-06-07): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: counted cycle 3 (review 2015-06-14): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: counted cycle 4 (review 2015-06-21): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: counted cycle 5 (review 2015-06-28): demanded, lost, stock-days, stockout flag | yes |
| naive c = 2: counted cycle 6 (review 2015-07-05): demanded, lost, stock-days, stockout flag | yes |
