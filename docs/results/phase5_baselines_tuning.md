# Phase 5 baselines on the tuning folds

Run 2026-09-20 at code commit `4f1b04a`. Tuning folds only (design section 12, `ml/folds.py`);
no test-window data was evaluated. Origins are Sundays: 4 folds x 12 origins x 300 series = 3600 rows per fold per horizon.
Series-level detail: `phase5_baselines_tuning.csv`, `phase5_baselines_tuning_by_segment.csv`.

WAPE is pooled over the fold; MASE is the mean over series of MAE / the series' in-sample naive-P MAE (zero-scale series excluded and
counted in the CSV); `mase_median` is the median over series, a robustness display added after seeing that one series dominates the F4 mean (see the note below); `share_y_le_forecast` is the share of actuals at or below the point forecast, i.e. the quantile level the forecast
implicitly sits at (point baselines have no intervals). Averages below are simple means over the four folds.

## Note on MASE in F4

The F4 mean MASE is dominated by one series, `FOODS_2_101_CA_2` (first sale 1,578 days after its first price, kept under the pre-registered rule).
It sold almost nothing before F4, so its naive scale is 0.03, and it started selling in F4: its naive MASE is 217. Without the five worst
series the F4 naive P = 14 mean falls from 1.70 to 0.93; the median is 0.87. The mean-over-series definition of design section 6 is unchanged;
the median column is an added diagnostic. Any model comparison should look at both.

## By fold

| fold   |   P | model   |    n |   wape |   mase |   mase_median |   mase_series |   mase_excluded |    mae |   rmse |   share_y_le_forecast |   fallback_share |
|:-------|----:|:--------|-----:|-------:|-------:|--------------:|--------------:|----------------:|-------:|-------:|----------------------:|-----------------:|
| F1     |   7 | naive   | 3600 |  0.434 |  0.943 |         0.872 |           298 |               2 |  4.001 |  7.391 |                 0.609 |                0 |
| F1     |   7 | ma28    | 3600 |  0.392 |  0.83  |         0.766 |           298 |               2 |  3.612 |  6.761 |                 0.6   |                0 |
| F1     |   7 | snaive  | 3600 |  0.704 |  1.481 |         1.199 |           298 |               2 |  6.494 | 11.979 |                 0.632 |                0 |
| F1     |  10 | naive   | 3600 |  0.415 |  0.945 |         0.865 |           298 |               2 |  5.275 | 10.273 |                 0.655 |                0 |
| F1     |  10 | ma28    | 3600 |  0.368 |  0.826 |         0.736 |           298 |               2 |  4.677 |  9.112 |                 0.609 |                0 |
| F1     |  10 | snaive  | 3600 |  0.649 |  1.45  |         1.131 |           298 |               2 |  8.258 | 15.37  |                 0.633 |                0 |
| F1     |  14 | naive   | 3600 |  0.354 |  0.922 |         0.864 |           298 |               2 |  6.481 | 12.676 |                 0.598 |                0 |
| F1     |  14 | ma28    | 3600 |  0.337 |  0.856 |         0.777 |           298 |               2 |  6.165 | 12.04  |                 0.584 |                0 |
| F1     |  14 | snaive  | 3600 |  0.607 |  1.551 |         1.2   |           298 |               2 | 11.116 | 20.794 |                 0.624 |                0 |
| F2     |   7 | naive   | 3600 |  0.477 |  1.005 |         0.896 |           298 |               2 |  4.386 | 10.227 |                 0.611 |                0 |
| F2     |   7 | ma28    | 3600 |  0.428 |  0.899 |         0.745 |           298 |               2 |  3.937 | 10.213 |                 0.61  |                0 |
| F2     |   7 | snaive  | 3600 |  0.737 |  1.481 |         1.177 |           298 |               2 |  6.77  | 14.917 |                 0.636 |                0 |
| F2     |  10 | naive   | 3600 |  0.442 |  0.984 |         0.889 |           298 |               2 |  5.646 | 12.739 |                 0.646 |                0 |
| F2     |  10 | ma28    | 3600 |  0.398 |  0.884 |         0.732 |           298 |               2 |  5.085 | 12.25  |                 0.609 |                0 |
| F2     |  10 | snaive  | 3600 |  0.689 |  1.472 |         1.177 |           298 |               2 |  8.806 | 18.686 |                 0.621 |                0 |
| F2     |  14 | naive   | 3600 |  0.389 |  0.982 |         0.827 |           298 |               2 |  7.147 | 17.335 |                 0.587 |                0 |
| F2     |  14 | ma28    | 3600 |  0.371 |  0.92  |         0.75  |           298 |               2 |  6.821 | 17.363 |                 0.585 |                0 |
| F2     |  14 | snaive  | 3600 |  0.662 |  1.597 |         1.253 |           298 |               2 | 12.168 | 26.232 |                 0.623 |                0 |
| F3     |   7 | naive   | 3600 |  0.475 |  1.041 |         0.87  |           299 |               1 |  4.181 |  8.521 |                 0.608 |                0 |
| F3     |   7 | ma28    | 3600 |  0.418 |  0.888 |         0.755 |           299 |               1 |  3.68  |  7.592 |                 0.575 |                0 |
| F3     |   7 | snaive  | 3600 |  0.817 |  1.538 |         1.227 |           299 |               1 |  7.19  | 14.841 |                 0.592 |                0 |
| F3     |  10 | naive   | 3600 |  0.442 |  1.029 |         0.884 |           299 |               1 |  5.408 | 11.907 |                 0.635 |                0 |
| F3     |  10 | ma28    | 3600 |  0.391 |  0.887 |         0.758 |           299 |               1 |  4.787 | 10.266 |                 0.572 |                0 |
| F3     |  10 | snaive  | 3600 |  0.764 |  1.538 |         1.205 |           299 |               1 |  9.355 | 19.559 |                 0.581 |                0 |
| F3     |  14 | naive   | 3600 |  0.378 |  1.005 |         0.861 |           299 |               1 |  6.696 | 14.485 |                 0.57  |                0 |
| F3     |  14 | ma28    | 3600 |  0.351 |  0.923 |         0.78  |           299 |               1 |  6.216 | 13.027 |                 0.551 |                0 |
| F3     |  14 | snaive  | 3600 |  0.727 |  1.675 |         1.295 |           299 |               1 | 12.878 | 26.869 |                 0.568 |                0 |
| F4     |   7 | naive   | 3600 |  0.428 |  1.463 |         0.908 |           300 |               0 |  3.921 |  7.646 |                 0.602 |                0 |
| F4     |   7 | ma28    | 3600 |  0.411 |  1.331 |         0.81  |           300 |               0 |  3.765 |  7.809 |                 0.565 |                0 |
| F4     |   7 | snaive  | 3600 |  0.819 |  2.865 |         1.297 |           300 |               0 |  7.51  | 15.869 |                 0.564 |                0 |
| F4     |  10 | naive   | 3600 |  0.408 |  1.492 |         0.912 |           300 |               0 |  5.168 | 10.574 |                 0.635 |                0 |
| F4     |  10 | ma28    | 3600 |  0.389 |  1.463 |         0.782 |           300 |               0 |  4.923 | 10.714 |                 0.569 |                0 |
| F4     |  10 | snaive  | 3600 |  0.777 |  3.404 |         1.272 |           300 |               0 |  9.829 | 21.291 |                 0.549 |                0 |
| F4     |  14 | naive   | 3600 |  0.363 |  1.701 |         0.871 |           300 |               0 |  6.648 | 14.396 |                 0.581 |                0 |
| F4     |  14 | ma28    | 3600 |  0.368 |  1.882 |         0.834 |           300 |               0 |  6.733 | 15.248 |                 0.544 |                0 |
| F4     |  14 | snaive  | 3600 |  0.741 |  4.402 |         1.33  |           300 |               0 | 13.571 | 30.031 |                 0.531 |                0 |

## Mean over the four folds

|                |   wape |   mase |   mase_median |    mae |   rmse |   share_y_le_forecast |
|:---------------|-------:|-------:|--------------:|-------:|-------:|----------------------:|
| (7, 'ma28')    |  0.412 |  0.987 |         0.769 |  3.748 |  8.094 |                 0.588 |
| (7, 'naive')   |  0.454 |  1.113 |         0.886 |  4.122 |  8.446 |                 0.607 |
| (7, 'snaive')  |  0.769 |  1.841 |         1.225 |  6.991 | 14.401 |                 0.606 |
| (10, 'ma28')   |  0.386 |  1.015 |         0.752 |  4.868 | 10.586 |                 0.59  |
| (10, 'naive')  |  0.427 |  1.112 |         0.888 |  5.374 | 11.373 |                 0.643 |
| (10, 'snaive') |  0.72  |  1.966 |         1.196 |  9.062 | 18.726 |                 0.596 |
| (14, 'ma28')   |  0.356 |  1.145 |         0.785 |  6.484 | 14.42  |                 0.566 |
| (14, 'naive')  |  0.371 |  1.152 |         0.856 |  6.743 | 14.723 |                 0.584 |
| (14, 'snaive') |  0.684 |  2.306 |         1.269 | 12.433 | 25.982 |                 0.587 |

## WAPE by velocity segment (mean over folds)

|                |   low |   mid |   high |
|:---------------|------:|------:|-------:|
| (7, 'ma28')    | 0.689 | 0.477 |  0.367 |
| (7, 'naive')   | 0.811 | 0.55  |  0.391 |
| (7, 'snaive')  | 1.022 | 0.803 |  0.736 |
| (10, 'ma28')   | 0.632 | 0.437 |  0.349 |
| (10, 'naive')  | 0.736 | 0.496 |  0.377 |
| (10, 'snaive') | 0.937 | 0.739 |  0.695 |
| (14, 'ma28')   | 0.558 | 0.394 |  0.326 |
| (14, 'naive')  | 0.61  | 0.417 |  0.334 |
| (14, 'snaive') | 0.857 | 0.687 |  0.668 |
