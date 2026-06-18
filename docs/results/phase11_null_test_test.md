# Shuffled-demand null test (test window, report-only)

Run 2026-09-21. Base case L = 3 (P = 10), 300 series. `cs90`, `cs95`, `fill90`: the quantile policy's cycle service at alpha 0.90 and 0.95 and its fill rate at 0.90; `red_b3a`, `red_b2`: comparator-anchored mean
inventory reduction of the quantile policy against B3a and B2 / B2-sqrt (means over the settings that matched; NaN draws are dropped from the means). Forecasts are untouched, only the realised demand is replaced.

| demand                                         |   draws |   cs90 mean |   cs95 mean |   fill90 mean |   red_b3a mean |   red_b2 mean | red_b3a 2.5%-97.5% over draws   | draws with red_b3a below the real value   |
|:-----------------------------------------------|--------:|------------:|------------:|--------------:|---------------:|--------------:|:--------------------------------|:------------------------------------------|
| real demand                                    |       1 |      0.9405 |      0.9707 |        0.9611 |         0.1601 |        0.2947 | -                               | -                                         |
| BLOCK (weekly blocks permuted within series)   |      20 |      0.9108 |      0.9483 |        0.9317 |         0.2699 |        0.426  | [22.6%, 30.3%]                  | 0 of 20                                   |
| SWAP (each series gets another series' demand) |      20 |      0.7115 |      0.7492 |        0.5706 |         0.0754 |        0.372  | [5.3%, 9.6%]                    | 20 of 20                                  |
