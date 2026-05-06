# Phase 7 quantile forecasts on the tuning folds

Run 2026-09-21 at code commit `45c581d` (uncommitted changes in ml/ or tests/). Tuning folds only; nothing evaluated on the test window (touch log unchanged). One model version per fold and horizon (fit set ends at c - 84 days; 12 Sunday evaluation origins; the reserved calibration window feeds sigma and conformal offsets only). Rules: design.md section 12, Phase 7 pre-registration and its amendment. Scaled pinball = per-series mean pinball / naive-P scale, averaged over the six quantiles; 'mean' and 'median' are over series. Means are over the 12 fold x horizon cells unless stated.

## Step A: scale floor (XGBoost quantile, depth 3 / min_child_weight 30, normalised)

|                  |   mean scaled pinball |
|:-----------------|----------------------:|
| 0.0357 units/day |                0.2739 |
| 0.0714 units/day |                0.2691 |
| 0.1429 units/day |                0.2664 |

Chosen: **0.1429** (rule: within 0.5% relative of the best is tied and goes to 1/14).

## Step B: config search (mean scaled pinball, lower is better)

| variant   | config                                   |   mean_sp |   median_sp |   F2_mean_sp |   rounds_hit_cap | selected   |
|:----------|:-----------------------------------------|----------:|------------:|-------------:|-----------------:|:-----------|
| raw       | {"max_depth": 3, "min_child_weight": 30} |    0.2873 |      0.2095 |       0.2553 |               12 |            |
| raw       | {"max_depth": 3, "min_child_weight": 5}  |    0.2874 |      0.2098 |       0.2555 |               12 |            |
| raw       | {"max_depth": 6, "min_child_weight": 30} |    0.2839 |      0.2049 |       0.2506 |                9 | yes        |
| raw       | {"max_depth": 6, "min_child_weight": 5}  |    0.2854 |      0.206  |       0.2522 |               10 |            |
| norm      | {"max_depth": 3, "min_child_weight": 30} |    0.2664 |      0.1909 |       0.2385 |               12 |            |
| norm      | {"max_depth": 3, "min_child_weight": 5}  |    0.2668 |      0.1911 |       0.239  |               12 |            |
| norm      | {"max_depth": 6, "min_child_weight": 30} |    0.2639 |      0.1861 |       0.2315 |                5 | yes        |
| norm      | {"max_depth": 6, "min_child_weight": 5}  |    0.2653 |      0.1877 |       0.2335 |                4 |            |

## Step C: normalisation decision (both mean and median must win by >= 1% relative)

| variant    |   mean_sp |   median_sp |   F2_mean_sp |   sp_low |   sp_mid |   sp_high |   WAPE_of_median |   MASE_of_median |
|:-----------|----------:|------------:|-------------:|---------:|---------:|----------:|-----------------:|-----------------:|
| raw        |    0.2839 |      0.2049 |       0.2506 |   0.4018 |   0.2359 |    0.2159 |           0.3572 |            0.994 |
| normalised |    0.2639 |      0.1861 |       0.2315 |   0.3564 |   0.2252 |    0.2116 |           0.3579 |            0.993 |

Relative change of normalised vs raw: mean -7.04%, median -9.19%. **Adopted variant: norm.**

## Step D: feature-group ablations at the adopted variant (norm, {"max_depth": 6, "min_child_weight": 30})

Groups `ev` (30 named-event indicators) and `xs` (cross-store zero run) are added only if they lower mean scaled pinball by >= 0.5% relative; the other arms are report-only sensitivities (ids, year-ago flipped, future price removed).

| arm   |   mean_sp |   median_sp |   F2_mean_sp | vs_base   |   WAPE_of_median | decision   |
|:------|----------:|------------:|-------------:|:----------|-----------------:|:-----------|
| base  |    0.2639 |      0.1861 |       0.2315 | +0.00%    |           0.3579 |            |
| ev    |    0.2628 |      0.1862 |       0.2317 | -0.42%    |           0.358  | do not add |
| xs    |    0.2639 |      0.1855 |       0.232  | -0.02%    |           0.3585 | do not add |

Final shared feature set for both policies: base set of the norm variant + `base`.

## Step E: scaled pinball against the point-policy benchmarks (mean over 12 cells; ratio below 1 is better)

| method                                                       |   mean_sp |   median_sp |   sp_low |   sp_mid |   sp_high |   sp_0.8 |   sp_0.9 |   sp_0.95 |   sp_0.99 |   interval80 |   ratio_mean_to_B2 |   ratio_mean_to_B1 |
|:-------------------------------------------------------------|----------:|------------:|---------:|---------:|----------:|---------:|---------:|----------:|----------:|-------------:|-------------------:|-------------------:|
| XGBoost quantile (headline)                                  |    0.2639 |      0.1861 |   0.3564 |   0.2252 |    0.2116 |   0.4016 |   0.2693 |    0.1672 |    0.0621 |       0.8714 |             0.6393 |             0.624  |
| XGBoost quantile + conformal (variant)                       |    0.2659 |      0.185  |   0.3631 |   0.2246 |    0.2114 |   0.4086 |   0.2723 |    0.1725 |    0.0587 |       0.8802 |             0.6441 |             0.6287 |
| B1: MA-28 + normal sigma                                     |    0.423  |      0.2973 |   0.3902 |   0.3094 |    0.57   |   0.7399 |   0.5497 |    0.3591 |    0.115  |       0.9631 |             1.0245 |             1      |
| B2: XGBoost mean + normal sigma (point policy)               |    0.4128 |      0.2872 |   0.392  |   0.2957 |    0.5517 |   0.7121 |   0.5262 |    0.3407 |    0.1062 |       0.9619 |             1      |             0.976  |
| C: RF mean + normal sigma (comparison only)                  |    0.3974 |      0.2754 |   0.3741 |   0.285  |    0.5338 |   0.682  |   0.5038 |    0.3269 |    0.1024 |       0.9649 |             0.9625 |             0.9395 |
| B2': frozen Phase 6 raw XGBoost + normal sigma (report-only) |    0.4082 |      0.2874 |   0.3921 |   0.2898 |    0.5436 |   0.702  |   0.5193 |    0.3366 |    0.1048 |       0.9649 |             0.9887 |             0.9651 |

### F2 (holiday fold) only

| method                                                       |   mean_sp |   median_sp |   sp_0.8 |   sp_0.9 |   sp_0.95 |   sp_0.99 |
|:-------------------------------------------------------------|----------:|------------:|---------:|---------:|----------:|----------:|
| XGBoost quantile (headline)                                  |    0.2315 |      0.1821 |   0.3468 |   0.2468 |    0.1656 |    0.0661 |
| XGBoost quantile + conformal (variant)                       |    0.2307 |      0.179  |   0.3442 |   0.2457 |    0.1652 |    0.0654 |
| B1: MA-28 + normal sigma                                     |    0.4017 |      0.2947 |   0.7392 |   0.5472 |    0.3583 |    0.1207 |
| B2: XGBoost mean + normal sigma (point policy)               |    0.3858 |      0.2855 |   0.7009 |   0.5133 |    0.3331 |    0.109  |
| C: RF mean + normal sigma (comparison only)                  |    0.3764 |      0.2765 |   0.6816 |   0.5004 |    0.3247 |    0.1061 |
| B2': frozen Phase 6 raw XGBoost + normal sigma (report-only) |    0.3907 |      0.2923 |   0.7081 |   0.5195 |    0.3371 |    0.1093 |

### Mean scaled pinball by horizon

|   P |   XGBoost quantile (headline) |   XGBoost quantile + conformal (variant) |   B1: MA-28 + normal sigma |   B2: XGBoost mean + normal sigma (point policy) |   C: RF mean + normal sigma (comparison only) |   B2': frozen Phase 6 raw XGBoost + normal sigma (report-only) |
|----:|------------------------------:|-----------------------------------------:|---------------------------:|-------------------------------------------------:|----------------------------------------------:|---------------------------------------------------------------:|
|   7 |                        0.2464 |                                   0.2467 |                     0.3849 |                                           0.3739 |                                        0.3666 |                                                         0.3765 |
|  10 |                        0.2602 |                                   0.2624 |                     0.4124 |                                           0.4077 |                                        0.3925 |                                                         0.3991 |
|  14 |                        0.2851 |                                   0.2887 |                     0.4715 |                                           0.4568 |                                        0.4329 |                                                         0.449  |

### Coverage of ceil(q): quantile model (status counts are over fold x horizon cells: 12 all, 3 for F2)

|   alpha | scope     |   cov_hi |   cov_lo |   tie |   zero_share |   raw_cov |   under |   consistent |   over |
|--------:|:----------|---------:|---------:|------:|-------------:|----------:|--------:|-------------:|-------:|
|    0.8  | all cells |    0.851 |    0.791 | 0.06  |        0.002 |     0.793 |       0 |           12 |      0 |
|    0.8  | F2 only   |    0.855 |    0.802 | 0.053 |        0.004 |     0.806 |       0 |            3 |      0 |
|    0.9  | all cells |    0.922 |    0.89  | 0.032 |        0     |     0.89  |       0 |           12 |      0 |
|    0.9  | F2 only   |    0.925 |    0.894 | 0.031 |        0     |     0.894 |       0 |            3 |      0 |
|    0.95 | all cells |    0.961 |    0.944 | 0.017 |        0     |     0.944 |       0 |           12 |      0 |
|    0.95 | F2 only   |    0.959 |    0.945 | 0.014 |        0     |     0.945 |       0 |            3 |      0 |
|    0.99 | all cells |    0.996 |    0.994 | 0.002 |        0     |     0.994 |       0 |           12 |      0 |
|    0.99 | F2 only   |    0.994 |    0.992 | 0.001 |        0     |     0.992 |       0 |            3 |      0 |

### Coverage of ceil(q): quantile model + conformal (status counts are over fold x horizon cells: 12 all, 3 for F2)

|   alpha | scope     |   cov_hi |   cov_lo |   tie |   zero_share |   raw_cov |   under |   consistent |   over |
|--------:|:----------|---------:|---------:|------:|-------------:|----------:|--------:|-------------:|-------:|
|    0.8  | all cells |    0.856 |    0.796 | 0.061 |        0.003 |     0.798 |       0 |           12 |      0 |
|    0.8  | F2 only   |    0.848 |    0.791 | 0.056 |        0.005 |     0.796 |       0 |            3 |      0 |
|    0.9  | all cells |    0.931 |    0.899 | 0.032 |        0     |     0.899 |       0 |           12 |      0 |
|    0.9  | F2 only   |    0.922 |    0.889 | 0.033 |        0     |     0.889 |       0 |            3 |      0 |
|    0.95 | all cells |    0.965 |    0.949 | 0.016 |        0     |     0.949 |       0 |           12 |      0 |
|    0.95 | F2 only   |    0.955 |    0.939 | 0.016 |        0     |     0.939 |       0 |            3 |      0 |
|    0.99 | all cells |    0.993 |    0.991 | 0.003 |        0     |     0.991 |       0 |           12 |      0 |
|    0.99 | F2 only   |    0.989 |    0.986 | 0.002 |        0     |     0.986 |       0 |            3 |      0 |

### Coverage of ceil(q): B2 point policy (XGBoost mean + normal) (status counts are over fold x horizon cells: 12 all, 3 for F2)

|   alpha | scope     |   cov_hi |   cov_lo |   tie |   zero_share |   raw_cov |   under |   consistent |   over |
|--------:|:----------|---------:|---------:|------:|-------------:|----------:|--------:|-------------:|-------:|
|    0.8  | all cells |    0.952 |    0.925 | 0.027 |            0 |     0.925 |       0 |            0 |     12 |
|    0.8  | F2 only   |    0.953 |    0.922 | 0.031 |            0 |     0.922 |       0 |            0 |      3 |
|    0.9  | all cells |    0.972 |    0.957 | 0.016 |            0 |     0.957 |       0 |            0 |     12 |
|    0.9  | F2 only   |    0.974 |    0.957 | 0.017 |            0 |     0.957 |       0 |            0 |      3 |
|    0.95 | all cells |    0.981 |    0.97  | 0.01  |            0 |     0.97  |       0 |           12 |      0 |
|    0.95 | F2 only   |    0.981 |    0.972 | 0.009 |            0 |     0.972 |       0 |            3 |      0 |
|    0.99 | all cells |    0.989 |    0.984 | 0.005 |            0 |     0.984 |       0 |           12 |      0 |
|    0.99 | F2 only   |    0.988 |    0.983 | 0.005 |            0 |     0.983 |       0 |            3 |      0 |

### Coverage of ceil(q): B2' frozen Phase 6 raw XGBoost + normal (report-only) (status counts are over fold x horizon cells: 12 all, 3 for F2)

|   alpha | scope     |   cov_hi |   cov_lo |   tie |   zero_share |   raw_cov |   under |   consistent |   over |
|--------:|:----------|---------:|---------:|------:|-------------:|----------:|--------:|-------------:|-------:|
|    0.8  | all cells |    0.952 |    0.925 | 0.027 |            0 |     0.925 |       0 |            0 |     12 |
|    0.8  | F2 only   |    0.955 |    0.926 | 0.029 |            0 |     0.926 |       0 |            0 |      3 |
|    0.9  | all cells |    0.972 |    0.957 | 0.015 |            0 |     0.957 |       0 |            0 |     12 |
|    0.9  | F2 only   |    0.975 |    0.959 | 0.015 |            0 |     0.959 |       0 |            0 |      3 |
|    0.95 | all cells |    0.98  |    0.97  | 0.01  |            0 |     0.97  |       0 |           12 |      0 |
|    0.95 | F2 only   |    0.982 |    0.973 | 0.009 |            0 |     0.973 |       0 |            3 |      0 |
|    0.99 | all cells |    0.988 |    0.984 | 0.004 |            0 |     0.984 |       0 |           12 |      0 |
|    0.99 | F2 only   |    0.988 |    0.984 | 0.004 |            0 |     0.984 |       0 |            3 |      0 |

### Coverage by velocity segment: quantile model (mean over cells)

|   alpha | segment   |   cov_hi |   cov_lo |
|--------:|:----------|---------:|---------:|
|    0.8  | low       |    0.903 |    0.82  |
|    0.8  | mid       |    0.841 |    0.777 |
|    0.8  | high      |    0.809 |    0.776 |
|    0.9  | low       |    0.953 |    0.912 |
|    0.9  | mid       |    0.915 |    0.878 |
|    0.9  | high      |    0.898 |    0.88  |
|    0.95 | low       |    0.978 |    0.958 |
|    0.95 | mid       |    0.959 |    0.94  |
|    0.95 | high      |    0.946 |    0.935 |
|    0.99 | low       |    0.997 |    0.994 |
|    0.99 | mid       |    0.996 |    0.994 |
|    0.99 | high      |    0.994 |    0.993 |

### Conformal: pooled scores per segment and level

Cells x levels x segments: 144; not calibrated (too few scores): 0; thin-sample warnings (n < 5/(1-alpha)): 0; minimum n 1089, median n 1111.

### Model diagnostics

Cells whose early stopping hit the 1,000-round cap: 5 of 12. Mean share of rows with crossing quantiles before sorting: 0.000.
