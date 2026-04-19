# Phase 6 point forecasts on the tuning folds

Run 2026-09-20 at code commit `54fb2f4`. Tuning folds only; no test-window data was evaluated (touch log unchanged). One model version per fold and horizon: fit set ends at c - 84 days, evaluated on the fold's 12 Sunday origins. All rules were pre-registered in design.md section 12 before scoring.

**MA-28 is the primary benchmark.** `ratio` = model WAPE / MA-28 WAPE (below 1 is better). **No normalisation decision is made here**: both variants are reported side by side and carried into Phase 7, which decides on scaled pinball. Means are simple means over folds. Full per-fit results: `phase6_point_all_fits.csv`.

## Scale floor (XGBoost depth 6, min_child_weight 30, normalised, mean WAPE over 12 cells)

|                  |   mean WAPE |
|:-----------------|------------:|
| 0.0179 units/day |      0.5447 |
| 0.0357 units/day |      0.4415 |
| 0.0714 units/day |      0.4015 |

Chosen (provisional, Phase 7 may re-test): **0.0714**.

## Search results (mean WAPE over the 12 cells; F2 shown separately)

| family   | variant   | config                                    |   mean_wape |   F2_wape | selected   |
|:---------|:----------|:------------------------------------------|------------:|----------:|:-----------|
| lr       | raw       | {}                                        |      0.3854 |    0.3955 | yes        |
| lr       | norm      | {}                                        |      0.5268 |    0.5338 | yes        |
| rf       | raw       | {"max_depth": 10, "min_samples_leaf": 60} |      0.3721 |    0.3857 |            |
| rf       | raw       | {"max_depth": 10, "min_samples_leaf": 20} |      0.3693 |    0.382  | yes        |
| rf       | raw       | {"max_depth": 16, "min_samples_leaf": 60} |      0.3721 |    0.386  |            |
| rf       | raw       | {"max_depth": 16, "min_samples_leaf": 20} |      0.3692 |    0.3822 |            |
| rf       | norm      | {"max_depth": 10, "min_samples_leaf": 60} |      0.382  |    0.3975 |            |
| rf       | norm      | {"max_depth": 10, "min_samples_leaf": 20} |      0.3842 |    0.3984 |            |
| rf       | norm      | {"max_depth": 16, "min_samples_leaf": 60} |      0.3774 |    0.3912 | yes        |
| rf       | norm      | {"max_depth": 16, "min_samples_leaf": 20} |      0.378  |    0.389  |            |
| xgb      | raw       | {"max_depth": 3, "min_child_weight": 30}  |      0.3761 |    0.3855 | yes        |
| xgb      | raw       | {"max_depth": 3, "min_child_weight": 5}   |      0.3763 |    0.3846 |            |
| xgb      | raw       | {"max_depth": 6, "min_child_weight": 30}  |      0.3751 |    0.3782 |            |
| xgb      | raw       | {"max_depth": 6, "min_child_weight": 5}   |      0.376  |    0.3804 |            |
| xgb      | norm      | {"max_depth": 3, "min_child_weight": 30}  |      0.4023 |    0.4124 | yes        |
| xgb      | norm      | {"max_depth": 3, "min_child_weight": 5}   |      0.4022 |    0.4122 |            |
| xgb      | norm      | {"max_depth": 6, "min_child_weight": 30}  |      0.4015 |    0.4158 |            |
| xgb      | norm      | {"max_depth": 6, "min_child_weight": 5}   |      0.4145 |    0.437  |            |

## Variant: raw target

Selected configs: lr: no hyperparameters, rf: {'max_depth': 10, 'min_samples_leaf': 20}, xgb: {'max_depth': 3, 'min_child_weight': 30}.

### Mean over the four folds

|              |   wape |   mase |   mase_median |   share_y_le_forecast |   ratio_to_ma28 |
|:-------------|-------:|-------:|--------------:|----------------------:|----------------:|
| (7, 'lr')    |  0.413 |  0.995 |         0.789 |                 0.579 |           1.003 |
| (7, 'ma28')  |  0.412 |  0.987 |         0.769 |                 0.588 |           1     |
| (7, 'rf')    |  0.396 |  0.978 |         0.756 |                 0.596 |           0.962 |
| (7, 'xgb')   |  0.406 |  1     |         0.789 |                 0.596 |           0.985 |
| (10, 'lr')   |  0.384 |  1.023 |         0.783 |                 0.524 |           0.994 |
| (10, 'ma28') |  0.386 |  1.015 |         0.752 |                 0.59  |           1     |
| (10, 'rf')   |  0.37  |  1.012 |         0.733 |                 0.616 |           0.957 |
| (10, 'xgb')  |  0.374 |  1.015 |         0.758 |                 0.584 |           0.967 |
| (14, 'lr')   |  0.359 |  1.136 |         0.825 |                 0.542 |           1.006 |
| (14, 'ma28') |  0.356 |  1.145 |         0.785 |                 0.566 |           1     |
| (14, 'rf')   |  0.342 |  1.115 |         0.77  |                 0.584 |           0.958 |
| (14, 'xgb')  |  0.349 |  1.161 |         0.807 |                 0.578 |           0.978 |

### F2 (holiday fold) only

|              |   wape |   mase |   mase_median |   share_y_le_forecast |   ratio_to_ma28 |
|:-------------|-------:|-------:|--------------:|----------------------:|----------------:|
| (7, 'lr')    |  0.425 |  0.913 |         0.777 |                 0.595 |           0.993 |
| (7, 'ma28')  |  0.428 |  0.899 |         0.745 |                 0.61  |           1     |
| (7, 'rf')    |  0.414 |  0.895 |         0.747 |                 0.622 |           0.966 |
| (7, 'xgb')   |  0.419 |  0.921 |         0.774 |                 0.609 |           0.977 |
| (10, 'lr')   |  0.39  |  0.884 |         0.766 |                 0.549 |           0.979 |
| (10, 'ma28') |  0.398 |  0.884 |         0.732 |                 0.609 |           1     |
| (10, 'rf')   |  0.379 |  0.879 |         0.725 |                 0.643 |           0.953 |
| (10, 'xgb')  |  0.379 |  0.878 |         0.736 |                 0.603 |           0.952 |
| (14, 'lr')   |  0.372 |  0.958 |         0.819 |                 0.557 |           1.002 |
| (14, 'ma28') |  0.371 |  0.92  |         0.75  |                 0.585 |           1     |
| (14, 'rf')   |  0.353 |  0.912 |         0.762 |                 0.612 |           0.951 |
| (14, 'xgb')  |  0.359 |  0.944 |         0.794 |                 0.592 |           0.968 |

### WAPE by velocity segment (mean over folds)

|              |   wape_low |   wape_mid |   wape_high |
|:-------------|-----------:|-----------:|------------:|
| (7, 'lr')    |      0.716 |      0.484 |       0.364 |
| (7, 'ma28')  |      0.689 |      0.477 |       0.367 |
| (7, 'rf')    |      0.707 |      0.466 |       0.347 |
| (7, 'xgb')   |      0.752 |      0.473 |       0.353 |
| (10, 'lr')   |      0.661 |      0.438 |       0.343 |
| (10, 'ma28') |      0.632 |      0.437 |       0.349 |
| (10, 'rf')   |      0.655 |      0.424 |       0.327 |
| (10, 'xgb')  |      0.664 |      0.425 |       0.332 |
| (14, 'lr')   |      0.594 |      0.401 |       0.324 |
| (14, 'ma28') |      0.558 |      0.394 |       0.326 |
| (14, 'rf')   |      0.57  |      0.38  |       0.309 |
| (14, 'xgb')  |      0.605 |      0.389 |       0.313 |

### WAPE per fold

|              |    F1 |    F2 |    F3 |    F4 |
|:-------------|------:|------:|------:|------:|
| (7, 'lr')    | 0.393 | 0.425 | 0.428 | 0.408 |
| (7, 'ma28')  | 0.392 | 0.428 | 0.418 | 0.411 |
| (7, 'rf')    | 0.374 | 0.414 | 0.409 | 0.389 |
| (7, 'xgb')   | 0.386 | 0.419 | 0.421 | 0.398 |
| (10, 'lr')   | 0.363 | 0.39  | 0.396 | 0.388 |
| (10, 'ma28') | 0.368 | 0.398 | 0.391 | 0.389 |
| (10, 'rf')   | 0.353 | 0.379 | 0.38  | 0.368 |
| (10, 'xgb')  | 0.355 | 0.379 | 0.384 | 0.377 |
| (14, 'lr')   | 0.338 | 0.372 | 0.358 | 0.366 |
| (14, 'ma28') | 0.337 | 0.371 | 0.351 | 0.368 |
| (14, 'rf')   | 0.324 | 0.353 | 0.343 | 0.347 |
| (14, 'xgb')  | 0.331 | 0.359 | 0.347 | 0.357 |

## Variant: normalised target (floor 0.0714)

Selected configs: lr: no hyperparameters, rf: {'max_depth': 16, 'min_samples_leaf': 60}, xgb: {'max_depth': 3, 'min_child_weight': 30}.

### Mean over the four folds

|              |   wape |   mase |   mase_median |   share_y_le_forecast |   ratio_to_ma28 |
|:-------------|-------:|-------:|--------------:|----------------------:|----------------:|
| (7, 'lr')    |  0.501 |  1.103 |         0.842 |                 0.612 |           1.216 |
| (7, 'ma28')  |  0.412 |  0.987 |         0.769 |                 0.588 |           1     |
| (7, 'rf')    |  0.405 |  0.978 |         0.767 |                 0.604 |           0.982 |
| (7, 'xgb')   |  0.428 |  1.017 |         0.798 |                 0.608 |           1.038 |
| (10, 'lr')   |  0.521 |  1.223 |         0.848 |                 0.599 |           1.347 |
| (10, 'ma28') |  0.386 |  1.015 |         0.752 |                 0.59  |           1     |
| (10, 'rf')   |  0.377 |  1.007 |         0.742 |                 0.609 |           0.977 |
| (10, 'xgb')  |  0.403 |  1.052 |         0.776 |                 0.622 |           1.043 |
| (14, 'lr')   |  0.559 |  1.47  |         0.941 |                 0.581 |           1.567 |
| (14, 'ma28') |  0.356 |  1.145 |         0.785 |                 0.566 |           1     |
| (14, 'rf')   |  0.35  |  1.13  |         0.777 |                 0.59  |           0.982 |
| (14, 'xgb')  |  0.376 |  1.195 |         0.814 |                 0.595 |           1.054 |

### F2 (holiday fold) only

|              |   wape |   mase |   mase_median |   share_y_le_forecast |   ratio_to_ma28 |
|:-------------|-------:|-------:|--------------:|----------------------:|----------------:|
| (7, 'lr')    |  0.512 |  1.01  |         0.817 |                 0.64  |           1.195 |
| (7, 'ma28')  |  0.428 |  0.899 |         0.745 |                 0.61  |           1     |
| (7, 'rf')    |  0.422 |  0.908 |         0.751 |                 0.63  |           0.985 |
| (7, 'xgb')   |  0.437 |  0.929 |         0.766 |                 0.628 |           1.02  |
| (10, 'lr')   |  0.52  |  1.046 |         0.828 |                 0.619 |           1.308 |
| (10, 'ma28') |  0.398 |  0.884 |         0.732 |                 0.609 |           1     |
| (10, 'rf')   |  0.388 |  0.881 |         0.712 |                 0.631 |           0.975 |
| (10, 'xgb')  |  0.415 |  0.933 |         0.742 |                 0.643 |           1.044 |
| (14, 'lr')   |  0.569 |  1.211 |         0.927 |                 0.591 |           1.534 |
| (14, 'ma28') |  0.371 |  0.92  |         0.75  |                 0.585 |           1     |
| (14, 'rf')   |  0.364 |  0.92  |         0.761 |                 0.615 |           0.981 |
| (14, 'xgb')  |  0.385 |  0.963 |         0.77  |                 0.615 |           1.037 |

### WAPE by velocity segment (mean over folds)

|              |   wape_low |   wape_mid |   wape_high |
|:-------------|-----------:|-----------:|------------:|
| (7, 'lr')    |      0.73  |      0.534 |       0.47  |
| (7, 'ma28')  |      0.689 |      0.477 |       0.367 |
| (7, 'rf')    |      0.718 |      0.472 |       0.355 |
| (7, 'xgb')   |      0.734 |      0.486 |       0.381 |
| (10, 'lr')   |      0.667 |      0.52  |       0.507 |
| (10, 'ma28') |      0.632 |      0.437 |       0.349 |
| (10, 'rf')   |      0.658 |      0.427 |       0.337 |
| (10, 'xgb')  |      0.679 |      0.448 |       0.364 |
| (14, 'lr')   |      0.603 |      0.515 |       0.567 |
| (14, 'ma28') |      0.558 |      0.394 |       0.326 |
| (14, 'rf')   |      0.572 |      0.386 |       0.318 |
| (14, 'xgb')  |      0.588 |      0.403 |       0.348 |

### WAPE per fold

|              |    F1 |    F2 |    F3 |    F4 |
|:-------------|------:|------:|------:|------:|
| (7, 'lr')    | 0.496 | 0.512 | 0.504 | 0.493 |
| (7, 'ma28')  | 0.392 | 0.428 | 0.418 | 0.411 |
| (7, 'rf')    | 0.386 | 0.422 | 0.414 | 0.398 |
| (7, 'xgb')   | 0.421 | 0.437 | 0.437 | 0.416 |
| (10, 'lr')   | 0.526 | 0.52  | 0.517 | 0.519 |
| (10, 'ma28') | 0.368 | 0.398 | 0.391 | 0.389 |
| (10, 'rf')   | 0.362 | 0.388 | 0.385 | 0.375 |
| (10, 'xgb')  | 0.4   | 0.415 | 0.409 | 0.389 |
| (14, 'lr')   | 0.561 | 0.569 | 0.544 | 0.561 |
| (14, 'ma28') | 0.337 | 0.371 | 0.351 | 0.368 |
| (14, 'rf')   | 0.333 | 0.364 | 0.347 | 0.356 |
| (14, 'xgb')  | 0.377 | 0.385 | 0.365 | 0.377 |

## Ablations at each selected config (mean over the 12 cells; F2 separately)

| family   | variant   | arm       |   mean_wape |   F2_wape |   mean_mase |   mean_mase_median |
|:---------|:----------|:----------|------------:|----------:|------------:|-------------------:|
| lr       | raw       | all       |      0.3854 |    0.3955 |       1.051 |              0.799 |
| lr       | raw       | no_sum364 |      0.3856 |    0.396  |       1.051 |              0.8   |
| lr       | norm      | all       |      0.5268 |    0.5338 |       1.265 |              0.877 |
| lr       | norm      | no_sum364 |      0.5263 |    0.533  |       1.267 |              0.875 |
| rf       | raw       | all       |      0.3693 |    0.382  |       1.035 |              0.753 |
| rf       | raw       | no_sum364 |      0.3702 |    0.3835 |       1.035 |              0.761 |
| rf       | raw       | no_item   |      0.3692 |    0.3816 |       1.033 |              0.754 |
| rf       | raw       | no_ids    |      0.3695 |    0.3819 |       1.037 |              0.754 |
| rf       | norm      | all       |      0.3774 |    0.3912 |       1.038 |              0.762 |
| rf       | norm      | no_sum364 |      0.3772 |    0.3917 |       1.04  |              0.768 |
| rf       | norm      | no_item   |      0.3773 |    0.3893 |       1.042 |              0.762 |
| rf       | norm      | no_ids    |      0.3772 |    0.3903 |       1.04  |              0.764 |
| xgb      | raw       | all       |      0.3761 |    0.3855 |       1.059 |              0.784 |
| xgb      | raw       | no_sum364 |      0.3774 |    0.3898 |       1.062 |              0.792 |
| xgb      | raw       | no_item   |      0.3783 |    0.3904 |       1.072 |              0.792 |
| xgb      | raw       | no_ids    |      0.3775 |    0.3891 |       1.07  |              0.793 |
| xgb      | norm      | all       |      0.4023 |    0.4124 |       1.088 |              0.796 |
| xgb      | norm      | no_sum364 |      0.4065 |    0.4196 |       1.094 |              0.805 |
| xgb      | norm      | no_item   |      0.4015 |    0.4174 |       1.091 |              0.791 |
| xgb      | norm      | no_ids    |      0.4028 |    0.4125 |       1.095 |              0.794 |

### Decisions under the pre-registered rules

| decision           | variant   |   with_feature |   without | rule                                              | outcome   |   no_item |
|:-------------------|:----------|---------------:|----------:|:--------------------------------------------------|:----------|----------:|
| year-ago sum_364_P | raw       |         0.3761 |    0.3774 | keep iff XGBoost gain >= 0.003                    | drop      |  nan      |
| ids (rf)           | raw       |         0.3693 |    0.3695 | fewer ids preferred unless more ids gain >= 0.003 | no_ids    |    0.3692 |
| ids (xgb)          | raw       |         0.3761 |    0.3775 | fewer ids preferred unless more ids gain >= 0.003 | no_ids    |    0.3783 |
| year-ago sum_364_P | norm      |         0.4023 |    0.4065 | keep iff XGBoost gain >= 0.003                    | keep      |  nan      |
| ids (rf)           | norm      |         0.3774 |    0.3772 | fewer ids preferred unless more ids gain >= 0.003 | no_ids    |    0.3773 |
| ids (xgb)          | norm      |         0.4023 |    0.4028 | fewer ids preferred unless more ids gain >= 0.003 | no_ids    |    0.4015 |
