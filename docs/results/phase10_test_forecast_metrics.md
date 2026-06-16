# Forecast accuracy on the test window (phase 10, made once)

Run 2026-09-21. The primary replay reported inventory and service results only; this file is the test-window forecast accuracy, on the same forecast tables. **Only these numbers are quoted for forecast accuracy.**
25 counted review dates (22 Nov 2015 to 8 May 2016) x 300 series; model versions v1 to v4 (v0 served only the warm-up); each version is one cell and values are means over cells, as in Phase 7. Scaled pinball is over the six alphas (0.10, 0.50, 0.80, 0.90, 0.95, 0.99) with each series scaled by its naive-P error before the version's cutoff. Coverage is of ceil(q) (cov_hi = share of demand at or below it; the status compares it with the nominal level, tolerance 0.03). B3a and B3c are post-hoc benchmarks; B2 is the pre-specified point policy; the framing rule applies. The item bootstrap reflects which items were sampled, not variation between periods.

## Horizon 10 (base case, lead time 3)

### Quantile forecasts, all rows (B3a first)

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95     |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:----------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.270571 |                             0.222362 |             0.336994 |     0.842604 | consistent   |     0.918542 | consistent   |      0.956875 | consistent      |      0.99125  | consistent    |
| Quantile model                                              |               0.229612 |                             0.187703 |             0.318859 |     0.843438 | consistent   |     0.925938 | consistent   |      0.96     | consistent      |      0.994479 | consistent    |
| B3c (post-hoc): MA-28 + empirical residual quantiles        |               0.288706 |                             0.234176 |             0.347441 |     0.839583 | consistent   |     0.914062 | consistent   |      0.954896 | consistent      |      0.990938 | consistent    |
| B2 (pre-specified): XGBoost mean + normal sigma             |               0.378341 |                             0.314631 |             0.343615 |     0.948125 | over         |     0.969375 | over         |      0.978646 | consistent/over |      0.986562 | consistent    |
| B1: MA-28 + normal sigma                                    |               0.397071 |                             0.331305 |             0.350378 |     0.937708 | over         |     0.963229 | over         |      0.971875 | consistent/over |      0.981979 | consistent    |

Quantile model against B3a: difference in mean scaled pinball -0.0410, 95% interval [-0.0504, -0.0337] over items; lower in 4 of 4 cells (versions).
Quantile model against B3c: -0.0591 [-0.0714, -0.0495].

### Point forecasts (WAPE, mean over cells; the pipeline's mean forecast is the normalised-target XGBoost)

|                           |   WAPE |
|:--------------------------|-------:|
| MA-28                     | 0.3504 |
| XGBoost mean (normalised) | 0.3436 |
| Quantile model median     | 0.3189 |

### By period (peak: cycles starting on or before 2016-01-03; rest: later)

**Peak**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.278596 |                             0.217159 |             0.362207 |     0.861667 | consistent   |     0.928889 | consistent   |      0.965556 | consistent    |      0.990556 | consistent    |
| Quantile model                                              |               0.245169 |                             0.186831 |             0.34746  |     0.854444 | consistent   |     0.918333 | consistent   |      0.958889 | consistent    |      0.990556 | consistent    |

**Rest**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8      |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:----------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.267707 |                             0.215481 |             0.336628 |     0.844479 | consistent/over |     0.91875  | consistent   |      0.956146 | consistent    |      0.991771 | consistent    |
| Quantile model                                              |               0.225761 |                             0.180073 |             0.317356 |     0.843854 | consistent      |     0.929375 | consistent   |      0.960521 | consistent    |      0.995    | consistent    |

### By velocity segment

**high**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9       |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-----------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.240694 |                             0.21053  |             0.289323 |     0.805    | consistent   |     0.896875 | consistent/under |      0.942813 | consistent    |      0.985    | consistent    |
| Quantile model                                              |               0.196817 |                             0.169176 |             0.273349 |     0.815312 | consistent   |     0.904062 | consistent       |      0.948438 | consistent    |      0.992188 | consistent    |

**mid**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.245521 |                             0.217535 |             0.39776  |     0.863552 | consistent   |     0.927599 | consistent   |      0.965037 | consistent    |      0.994121 | consistent    |
| Quantile model                                              |               0.21049  |                             0.182912 |             0.378351 |     0.848391 | consistent   |     0.930384 | consistent   |      0.966584 | consistent    |      0.99505  | consistent    |

**low**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.326306 |                             0.241126 |             0.554377 |     0.859217 | consistent   |     0.931187 | consistent   |      0.962753 | consistent    |      0.994634 | consistent    |
| Quantile model                                              |               0.282246 |                             0.212485 |             0.52446  |     0.866793 | consistent   |     0.943497 | consistent   |      0.964962 | consistent    |      0.996212 | consistent    |

## Horizon 14 (lead time 7 sensitivity)

### Quantile forecasts, all rows (B3a first)

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9      |   cov_hi 0.95 | status 0.95     |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:----------------|--------------:|:----------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.298535 |                             0.2416   |             0.312703 |     0.824479 | consistent   |     0.908958 | consistent      |      0.955417 | consistent      |      0.991667 | consistent    |
| Quantile model                                              |               0.244363 |                             0.196097 |             0.295305 |     0.823229 | consistent   |     0.90375  | consistent      |      0.952604 | consistent      |      0.992917 | consistent    |
| B3c (post-hoc): MA-28 + empirical residual quantiles        |               0.317733 |                             0.256079 |             0.319351 |     0.826042 | consistent   |     0.908542 | consistent      |      0.950417 | consistent      |      0.990938 | consistent    |
| B2 (pre-specified): XGBoost mean + normal sigma             |               0.438055 |                             0.355199 |             0.31456  |     0.945833 | over         |     0.967187 | over            |      0.974583 | consistent/over |      0.983542 | consistent    |
| B1: MA-28 + normal sigma                                    |               0.462721 |                             0.378633 |             0.319936 |     0.934792 | over         |     0.958542 | consistent/over |      0.968646 | consistent/over |      0.978333 | consistent    |

Quantile model against B3a: difference in mean scaled pinball -0.0542, 95% interval [-0.0642, -0.0460] over items; lower in 4 of 4 cells (versions).
Quantile model against B3c: -0.0734 [-0.0854, -0.0635].

### Point forecasts (WAPE, mean over cells; the pipeline's mean forecast is the normalised-target XGBoost)

|                           |   WAPE |
|:--------------------------|-------:|
| MA-28                     | 0.3199 |
| XGBoost mean (normalised) | 0.3146 |
| Quantile model median     | 0.2953 |

### By period (peak: cycles starting on or before 2016-01-03; rest: later)

**Peak**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.289447 |                             0.227279 |             0.329352 |     0.859333 | consistent   |     0.929333 | consistent   |         0.964 | consistent    |         0.988 | consistent    |
| Quantile model                                              |               0.249285 |                             0.197595 |             0.319656 |     0.841333 | consistent   |     0.920667 | consistent   |         0.954 | consistent    |         0.994 | consistent    |

**Rest**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.297486 |                             0.238621 |             0.31233  |     0.825799 | consistent   |     0.908889 | consistent   |      0.955139 | consistent    |      0.9925   | consistent    |
| Quantile model                                              |               0.244393 |                             0.191506 |             0.292228 |     0.826146 | consistent   |     0.903993 | consistent   |      0.954236 | consistent    |      0.992812 | consistent    |

### By velocity segment

**high**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8            |   cov_hi 0.9 | status 0.9       |   cov_hi 0.95 | status 0.95      |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:----------------------|-------------:|:-----------------|--------------:|:-----------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.284641 |                             0.241041 |             0.275857 |     0.785937 | consistent/over/under |       0.8825 | consistent/under |      0.935625 | consistent/under |      0.985937 | consistent    |
| Quantile model                                              |               0.218264 |                             0.178175 |             0.260308 |     0.783125 | consistent/under      |       0.89   | consistent/under |      0.9375   | consistent       |      0.989375 | consistent    |

**mid**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9   |   cov_hi 0.95 | status 0.95     |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-------------|--------------:|:----------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.272479 |                             0.237715 |             0.352186 |     0.83323  | consistent   |     0.921411 | consistent   |      0.967203 | consistent/over |      0.994121 | consistent    |
| Quantile model                                              |               0.223928 |                             0.187021 |             0.335371 |     0.836634 | consistent   |     0.898824 | consistent   |      0.953589 | consistent      |      0.992265 | consistent    |

**low**

|                                                             |   scaled pinball, mean |   scaled pinball, median over series |   WAPE of the median |   cov_hi 0.8 | status 0.8   |   cov_hi 0.9 | status 0.9       |   cov_hi 0.95 | status 0.95   |   cov_hi 0.99 | status 0.99   |
|:------------------------------------------------------------|-----------------------:|-------------------------------------:|---------------------:|-------------:|:-------------|-------------:|:-----------------|--------------:|:--------------|--------------:|:--------------|
| B3a (post-hoc): XGBoost mean + empirical residual quantiles |               0.339152 |                             0.253349 |             0.496045 |     0.854482 | consistent   |     0.92298  | consistent       |      0.963384 | consistent    |      0.994949 | consistent    |
| Quantile model                                              |               0.291574 |                             0.218106 |             0.46505  |     0.850063 | consistent   |     0.922664 | consistent/under |      0.966856 | consistent    |      0.997159 | consistent    |
