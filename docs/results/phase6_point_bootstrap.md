# Phase 6: cells won and item-cluster bootstrap against MA-28

Raw target, frozen configs, tuning folds only (report-only, added after Phase 6 closed). `wape_diff` = pooled WAPE of the model minus MA-28 (negative = model better); 95% percentile interval from 10000 resamples of the 100 items (each item's three store-series resampled together; seed 0). The interval covers item sampling only, not period-to-period variability (one set of folds), so it understates uncertainty about other periods. `cells_won_of_12` counts fold x horizon cells where the model's WAPE is below MA-28's.

| family   |   cells_won_of_12 | group            |   wape_diff |   ci_low |   ci_high | excludes_zero   |
|:---------|------------------:|:-----------------|------------:|---------:|----------:|:----------------|
| lr       |                 6 | all 12 cells     |      0.0004 |  -0.009  |    0.0118 | no              |
| lr       |                 6 | P = 7 (4 folds)  |      0.0011 |  -0.0085 |    0.0123 | no              |
| lr       |                 6 | P = 10 (4 folds) |     -0.0022 |  -0.0116 |    0.0087 | no              |
| lr       |                 6 | P = 14 (4 folds) |      0.002  |  -0.0079 |    0.0142 | no              |
| lr       |                 6 | F2, all horizons |     -0.003  |  -0.0179 |    0.0135 | no              |
| lr       |                 6 | F2, P = 7        |     -0.0031 |  -0.0183 |    0.0132 | no              |
| lr       |                 6 | F2, P = 14       |      0.0007 |  -0.0153 |    0.0185 | no              |
| rf       |                12 | all 12 cells     |     -0.0157 |  -0.0228 |   -0.0085 | yes             |
| rf       |                12 | P = 7 (4 folds)  |     -0.0158 |  -0.0237 |   -0.0083 | yes             |
| rf       |                12 | P = 10 (4 folds) |     -0.0166 |  -0.0244 |   -0.0086 | yes             |
| rf       |                12 | P = 14 (4 folds) |     -0.015  |  -0.0218 |   -0.008  | yes             |
| rf       |                12 | F2, all horizons |     -0.0175 |  -0.0287 |   -0.0052 | yes             |
| rf       |                12 | F2, P = 7        |     -0.0145 |  -0.026  |   -0.0028 | yes             |
| rf       |                12 | F2, P = 14       |     -0.0181 |  -0.0298 |   -0.0051 | yes             |
| xgb      |                11 | all 12 cells     |     -0.0091 |  -0.0164 |   -0.001  | yes             |
| xgb      |                11 | P = 7 (4 folds)  |     -0.0064 |  -0.014  |    0.0019 | no              |
| xgb      |                11 | P = 10 (4 folds) |     -0.0127 |  -0.0206 |   -0.0038 | yes             |
| xgb      |                11 | P = 14 (4 folds) |     -0.0079 |  -0.0151 |   -0      | yes             |
| xgb      |                11 | F2, all horizons |     -0.0137 |  -0.0258 |   -0.0004 | yes             |
| xgb      |                11 | F2, P = 7        |     -0.0097 |  -0.0229 |    0.0043 | no              |
| xgb      |                11 | F2, P = 14       |     -0.0119 |  -0.0246 |    0.0024 | no              |
