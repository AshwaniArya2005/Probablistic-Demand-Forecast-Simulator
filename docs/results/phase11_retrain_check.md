# Retraining one frozen version from scratch (tuning data only)

Run 2026-09-21 at commit `2785d21`. Development fold F4, horizon 10 (cutoff 2015-05-31, fit set ending 2015-03-08), the frozen Phase 7 recipe (normalised target, floor 0.1429, quantile XGBoost
max_depth 6 / min_child_weight 30, mean XGBoost max_depth 3 / min_child_weight 30), seed 0, 5 threads. Compared on the 3600 review-origin rows of the fold.

| comparison | maximum absolute difference (units of demand) | within tolerance 0.0001 |
|---|---:|---|
| new quantile fit vs the quantile forecasts in the forecast table the results were built from | 0 | yes |
| new mean fit vs the mean forecasts in that table | 0 | yes |
| new quantile fit vs the saved model file (hash recorded in `models/manifest.jsonl`) | 0 | yes |
| trees in the new quantile model vs the saved one | 6000 vs 6000 | yes |

**Result: the retrained version reproduces the frozen forecasts within the stated tolerance.** Scope: one cheap version on the same machine, thread count and library versions; it does not show
cross-machine bit-reproducibility or that every version reproduces, and the test-window versions were not retrained (that would need its own touch-log entry).
