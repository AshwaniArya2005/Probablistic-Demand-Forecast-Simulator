# Shuffled-demand null test (tuning folds F2 and F4, report-only)

Run 2026-09-21. Base case L = 3 (P = 10), 300 series. `cs90`, `cs95`, `fill90`: the quantile policy's cycle service at alpha 0.90 and 0.95 and its fill rate at 0.90; `red_b3a`, `red_b2`: comparator-anchored mean
inventory reduction of the quantile policy against B3a and B2 / B2-sqrt (means over the settings that matched; NaN draws are dropped from the means). Forecasts are untouched, only the realised demand is replaced.

| demand                                         |   draws |   cs90 mean |   cs95 mean |   fill90 mean |   red_b3a mean |   red_b2 mean | red_b3a 2.5%-97.5% over draws   | draws with red_b3a below the real value   |
|:-----------------------------------------------|--------:|------------:|------------:|--------------:|---------------:|--------------:|:--------------------------------|:------------------------------------------|
| real demand                                    |       1 |      0.9439 |      0.9701 |        0.9527 |         0.0393 |        0.3006 | -                               | -                                         |
| BLOCK (weekly blocks permuted within series)   |      20 |      0.9347 |      0.9693 |        0.9463 |         0.1299 |        0.3553 | [9.4%, 21.0%]                   | 0 of 20                                   |
| SWAP (each series gets another series' demand) |      20 |      0.721  |      0.7604 |        0.5431 |         0.0969 |        0.4976 | [6.5%, 14.8%]                   | 0 of 20                                   |
