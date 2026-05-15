# Phase 7: cells won, item-cluster bootstrap and the post-hoc B3 benchmarks

Report-only, tuning folds only (touch log unchanged). The quantile model is the frozen Phase 7 model (normalised target, floor 1/7, depth 6 / min_child_weight 30, base feature set); per-series values were recomputed and reproduce the cached Phase 7 cell values to 1e-9. `diff` = mean scaled pinball of the quantile model minus the benchmark (negative = quantile model better), 95% percentile interval from 10000 resamples of the 100 items (each item's three store-series together, seed 0); it reflects item sampling only, not period-to-period variability. `cells_won_of_12` counts fold x horizon cells where the quantile model's mean scaled pinball is lower. **B3 variants are post-hoc**: defined after the Phase 7 results were known (design.md section 12, 2026-09-21), they can change no decision.

## Mean scaled pinball by method (mean over 12 cells of the per-cell mean and median over series)

| method         |   mean |   median |
|:---------------|-------:|---------:|
| quantile_model | 0.2639 |   0.1861 |
| B1_normal      | 0.423  |   0.2973 |
| B2_normal      | 0.4128 |   0.2872 |
| B2raw_normal   | 0.4082 |   0.2874 |
| B3a_xgb        | 0.3025 |   0.2192 |
| B3b_raw_xgb    | 0.3069 |   0.2231 |
| B3c_ma28       | 0.3128 |   0.2347 |

## Coverage of ceil(q) for the post-hoc B3 methods (mean over cells; status counts over 12 cells)

| method      |   alpha |   cov_hi |   cov_lo |   under |   consistent |   over |
|:------------|--------:|---------:|---------:|--------:|-------------:|-------:|
| B3a_xgb     |    0.8  |    0.854 |    0.796 |       0 |           12 |      0 |
| B3a_xgb     |    0.9  |    0.928 |    0.899 |       0 |           12 |      0 |
| B3a_xgb     |    0.95 |    0.965 |    0.951 |       0 |           12 |      0 |
| B3a_xgb     |    0.99 |    0.993 |    0.991 |       0 |           12 |      0 |
| B3b_raw_xgb |    0.8  |    0.855 |    0.797 |       0 |           12 |      0 |
| B3b_raw_xgb |    0.9  |    0.929 |    0.9   |       0 |           12 |      0 |
| B3b_raw_xgb |    0.95 |    0.966 |    0.951 |       0 |           12 |      0 |
| B3b_raw_xgb |    0.99 |    0.993 |    0.991 |       0 |           12 |      0 |
| B3c_ma28    |    0.8  |    0.854 |    0.798 |       0 |           12 |      0 |
| B3c_ma28    |    0.9  |    0.929 |    0.9   |       0 |           12 |      0 |
| B3c_ma28    |    0.95 |    0.965 |    0.951 |       0 |           12 |      0 |
| B3c_ma28    |    0.99 |    0.993 |    0.991 |       0 |           12 |      0 |

## Cells won and bootstrap intervals

| benchmark                                                         |   cells_won_of_12 |   F2_cells_won_of_3 | group                   |    diff |   ci_low |   ci_high | excludes_zero   |
|:------------------------------------------------------------------|------------------:|--------------------:|:------------------------|--------:|---------:|----------:|:----------------|
| B1: MA-28 + normal sigma                                          |                12 |                   3 | all 12 cells            | -0.159  |  -0.1925 |   -0.1284 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | F2 (3 cells)            | -0.1702 |  -0.2144 |   -0.1296 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | P = 7 (4 folds)         | -0.1385 |  -0.1695 |   -0.1108 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | P = 10 (4 folds)        | -0.1522 |  -0.1868 |   -0.1203 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | P = 14 (4 folds)        | -0.1864 |  -0.2257 |   -0.1507 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | segment low (12 cells)  | -0.0339 |  -0.0446 |   -0.0229 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | segment mid (12 cells)  | -0.0842 |  -0.0983 |   -0.0717 | yes             |
| B1: MA-28 + normal sigma                                          |                12 |                   3 | segment high (12 cells) | -0.3584 |  -0.4121 |   -0.3084 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | all 12 cells            | -0.1489 |  -0.1805 |   -0.1201 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | F2 (3 cells)            | -0.1544 |  -0.1937 |   -0.1174 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | P = 7 (4 folds)         | -0.1275 |  -0.1555 |   -0.1025 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | P = 10 (4 folds)        | -0.1475 |  -0.1809 |   -0.1171 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | P = 14 (4 folds)        | -0.1717 |  -0.2083 |   -0.1387 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | segment low (12 cells)  | -0.0356 |  -0.0443 |   -0.0272 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | segment mid (12 cells)  | -0.0706 |  -0.082  |   -0.0602 | yes             |
| B2: XGBoost mean + normal sigma (point policy)                    |                12 |                   3 | segment high (12 cells) | -0.3401 |  -0.3906 |   -0.2928 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | all 12 cells            | -0.1443 |  -0.1747 |   -0.1164 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | F2 (3 cells)            | -0.1592 |  -0.1985 |   -0.1222 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | P = 7 (4 folds)         | -0.1301 |  -0.1577 |   -0.1054 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | P = 10 (4 folds)        | -0.1389 |  -0.1714 |   -0.1091 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | P = 14 (4 folds)        | -0.1638 |  -0.1998 |   -0.1315 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | segment low (12 cells)  | -0.0358 |  -0.053  |   -0.0054 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | segment mid (12 cells)  | -0.0646 |  -0.075  |   -0.0552 | yes             |
| B2': frozen Phase 6 raw XGBoost + normal sigma                    |                12 |                   3 | segment high (12 cells) | -0.3321 |  -0.3824 |   -0.285  | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | all 12 cells            | -0.0386 |  -0.0493 |   -0.0312 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | F2 (3 cells)            | -0.0289 |  -0.0347 |   -0.0238 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | P = 7 (4 folds)         | -0.0343 |  -0.0472 |   -0.0263 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | P = 10 (4 folds)        | -0.0337 |  -0.0385 |   -0.0292 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | P = 14 (4 folds)        | -0.0477 |  -0.0636 |   -0.0375 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | segment low (12 cells)  | -0.0342 |  -0.0651 |   -0.0181 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | segment mid (12 cells)  | -0.0354 |  -0.0418 |   -0.0298 | yes             |
| B3a (post-hoc): XGBoost mean + empirical residual quantiles       |                12 |                   3 | segment high (12 cells) | -0.0461 |  -0.0534 |   -0.0385 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | all 12 cells            | -0.043  |  -0.0602 |   -0.0327 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | F2 (3 cells)            | -0.0303 |  -0.0358 |   -0.0251 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | P = 7 (4 folds)         | -0.0411 |  -0.0636 |   -0.0282 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | P = 10 (4 folds)        | -0.0415 |  -0.0585 |   -0.0314 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | P = 14 (4 folds)        | -0.0463 |  -0.0582 |   -0.038  | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | segment low (12 cells)  | -0.052  |  -0.1072 |   -0.0256 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | segment mid (12 cells)  | -0.034  |  -0.0398 |   -0.0287 | yes             |
| B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles |                12 |                   3 | segment high (12 cells) | -0.0431 |  -0.051  |   -0.0354 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | all 12 cells            | -0.0488 |  -0.055  |   -0.0432 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | F2 (3 cells)            | -0.0426 |  -0.0518 |   -0.0347 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | P = 7 (4 folds)         | -0.0456 |  -0.0572 |   -0.0374 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | P = 10 (4 folds)        | -0.0429 |  -0.0504 |   -0.0338 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | P = 14 (4 folds)        | -0.0581 |  -0.0655 |   -0.0514 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | segment low (12 cells)  | -0.0372 |  -0.0469 |   -0.029  | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | segment mid (12 cells)  | -0.0496 |  -0.0586 |   -0.0417 | yes             |
| B3c (post-hoc): MA-28 + empirical residual quantiles              |                12 |                   3 | segment high (12 cells) | -0.0596 |  -0.0691 |   -0.0499 | yes             |
