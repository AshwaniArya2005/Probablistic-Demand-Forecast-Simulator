# Phase 9 planner quantities on the tuning folds

Run 2026-09-21 at code commit `3781317`. Report-only; **tuning folds only**, touch log unchanged. Definitions, thresholds and the constant-CV rule were fixed in docs/design.md
(Phase 9 pre-registration) before this run. Rows: 43200 series-origins (12 fold x horizon cells of Sunday review origins); reorder points use P in {10, 14}.

## 1. Constant-CV check (calibration windows; 4 scale bins per segment; accepted if max/min bin RMSE <= 1.5, else the sqrt(scale) fallback)

Count of fold x horizon cells (12 per segment) by the form the check selects:

| segment   |   constant-CV (both failed) |   sqrt-scale |
|:----------|----------------------------:|-------------:|
| low       |                           0 |           12 |
| mid       |                           6 |            6 |
| high      |                          12 |            0 |

Bin ratio of scale-normalised residuals (constant-CV form) and of the fallback, by segment (median and max over the 12 cells):

| segment   |   ('ratio_constant_cv', 'median') |   ('ratio_constant_cv', 'max') |   ('ratio_sqrt_scale', 'median') |   ('ratio_sqrt_scale', 'max') |
|:----------|----------------------------------:|-------------------------------:|---------------------------------:|------------------------------:|
| low       |                              2.09 |                           2.62 |                             1.23 |                          1.47 |
| mid       |                              3.84 |                           5.62 |                             1.55 |                          2.14 |
| high      |                             12.36 |                          16.28 |                             2.97 |                          4.19 |

## 2. Classic vs quantile reorder points (pooled over folds, P = 10 and 14; coverage = share of actual protection demand at or below the reorder point)

|                                       |   mean_rop |   mean_safety_stock |   coverage |
|:--------------------------------------|-----------:|--------------------:|-----------:|
| (0.8, 'classic (normal)')             |     44.376 |              28.212 |      0.944 |
| (0.8, 'post-hoc empirical residual')  |     21.451 |               5.287 |      0.848 |
| (0.8, 'quantile')                     |     20.011 |               4.136 |      0.844 |
| (0.9, 'classic (normal)')             |     58.871 |              42.708 |      0.969 |
| (0.9, 'post-hoc empirical residual')  |     26.334 |              10.171 |      0.925 |
| (0.9, 'quantile')                     |     23.567 |               7.505 |      0.917 |
| (0.95, 'classic (normal)')            |     70.839 |              54.676 |      0.979 |
| (0.95, 'post-hoc empirical residual') |     32.828 |              16.665 |      0.963 |
| (0.95, 'quantile')                    |     27.443 |              11.321 |      0.96  |
| (0.99, 'classic (normal)')            |     93.287 |              77.123 |      0.988 |
| (0.99, 'post-hoc empirical residual') |     92.982 |              76.819 |      0.992 |
| (0.99, 'quantile')                    |     48.193 |              32.036 |      0.996 |

Mean safety stock (units) by segment:

|                |   classic (normal) |   post-hoc empirical residual |   quantile |
|:---------------|-------------------:|------------------------------:|-----------:|
| (0.8, 'low')   |               2.85 |                          1.95 |       2.09 |
| (0.8, 'mid')   |               7.67 |                          4.06 |       3.15 |
| (0.8, 'high')  |              74.07 |                          9.83 |       7.16 |
| (0.9, 'low')   |               4.1  |                          3.96 |       3.53 |
| (0.9, 'mid')   |              11.43 |                          7.63 |       5.51 |
| (0.9, 'high')  |             112.51 |                         18.89 |      13.45 |
| (0.95, 'low')  |               5.12 |                          6.13 |       5.06 |
| (0.95, 'mid')  |              14.54 |                         12.52 |       8.16 |
| (0.95, 'high') |             144.27 |                         31.28 |      20.7  |
| (0.99, 'low')  |               7.03 |                         12.01 |      10.37 |
| (0.99, 'mid')  |              20.36 |                         37.26 |      20.36 |
| (0.99, 'high') |             203.84 |                        180.94 |      65.28 |

## 3. Stockout-risk label validity (reference stock positions IP = ceil(c x mu28 x P), c = 1, 1.5, 2; not a policy, only a check of what the labels mean)

|   c | label          |   share_of_origins |     n |   realised_stockout_freq |
|----:|:---------------|-------------------:|------:|-------------------------:|
| 1   | HIGH           |              0.262 |  7547 |                    0.45  |
| 1   | MEDIUM         |              0.734 | 21135 |                    0.304 |
| 1   | LOW            |              0.004 |   118 |                    0.119 |
| 1   | overstock flag |              0     |     0 |                  nan     |
| 1.5 | HIGH           |              0.102 |  2949 |                    0.309 |
| 1.5 | MEDIUM         |              0.455 | 13098 |                    0.185 |
| 1.5 | LOW            |              0.443 | 12753 |                    0.049 |
| 1.5 | overstock flag |              0     |     0 |                  nan     |
| 2   | HIGH           |              0.087 |  2515 |                    0.26  |
| 2   | MEDIUM         |              0.213 |  6139 |                    0.172 |
| 2   | LOW            |              0.7   | 20146 |                    0.021 |
| 2   | overstock flag |              0     |     0 |                  nan     |

Reading: the label is a description of the position against the model's demand distribution, not a calibrated probability (design.md, Phase 9 pre-registration, item 1).
