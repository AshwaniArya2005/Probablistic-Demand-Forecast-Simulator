# Phase 8 explanations on a tuning fold (F2, P = 14, service-level quantile 0.9)

Run 2026-09-21 at code commit `f9ede98`. **Tuning fold only**; no test-window origin is explained (`explain_rows` refuses them) and the touch log is unchanged. Model: the frozen Phase 7 final quantile model (normalised target, floor 1/7, depth 6 / min_child_weight 30, base feature set) fitted for this fold's version. Contributions are exact TreeSHAP values in units (ratio-space contributions times each row's scale) of the reported quantile; they describe **associations in the model, not causal effects**. Explained rows: every Sunday review origin of the fold (3600 series-origins).

## Exactness checks over every explained row

- Additivity (baseline + sum of contributions = the reported quantile before clipping at 0): max absolute gap 4.44e-04 units, max relative gap 5.04e-06, for the P50 and the P90 together (7200 values).
- Rows where the explained quantile came from another target's trees because XGBoost's internal sort changed the order: 89 of 3600 (2.472%); the fit's unsorted crossing share is 2.667%.
- Rows whose reported value is below 0 (floored at 0 in the forecast): 75.

## Share of the absolute contribution by theme (mean over rows)

| quantile   | segment   |   recent sales level |   zero-sales stretches |   same period last year |   price vs usual |   price level |   calendar |   events |   series age |   series identity |   mean_abs_total_units |   mean_baseline_units |   mean_forecast_units |
|:-----------|:----------|---------------------:|-----------------------:|------------------------:|-----------------:|--------------:|-----------:|---------:|-------------:|------------------:|-----------------------:|----------------------:|----------------------:|
| P50        | all       |                0.25  |                  0.252 |                   0.179 |            0.027 |         0.053 |      0.066 |    0.011 |        0.16  |                 0 |                  5.288 |                16.629 |                16.575 |
| P50        | low       |                0.292 |                  0.352 |                   0.135 |            0.044 |         0.048 |      0.049 |    0.012 |        0.068 |                 0 |                  1.552 |                 3.903 |                 3.589 |
| P50        | mid       |                0.284 |                  0.311 |                   0.113 |            0.042 |         0.042 |      0.071 |    0.014 |        0.123 |                 0 |                  2.985 |                10.004 |                 9.91  |
| P50        | high      |                0.236 |                  0.223 |                   0.202 |            0.021 |         0.057 |      0.068 |    0.01  |        0.183 |                 0 |                 11.312 |                35.919 |                36.163 |
| P90        | all       |                0.107 |                  0.527 |                   0.242 |            0.011 |         0.04  |      0.02  |    0.003 |        0.05  |                 0 |                 21.089 |                31.553 |                26.85  |
| P90        | low       |                0.223 |                  0.407 |                   0.146 |            0.031 |         0.077 |      0.026 |    0.006 |        0.085 |                 0 |                  3.766 |                 7.382 |                 8.028 |
| P90        | mid       |                0.219 |                  0.454 |                   0.158 |            0.029 |         0.058 |      0.037 |    0.005 |        0.04  |                 0 |                  6.97  |                18.919 |                16.922 |
| P90        | high      |                0.083 |                  0.546 |                   0.26  |            0.007 |         0.035 |      0.017 |    0.003 |        0.049 |                 0 |                 52.499 |                68.242 |                55.511 |

## Example explanations (chosen by a fixed rule, not by appeal)

**high velocity, P50 forecast near the segment's 50th percentile** (`FOODS_2_380_CA_1_evaluation`, origin 2015-01-11, velocity segment high):

- P50 forecast: 29.6 units over the next 14 days (typical 27.0). The recent sales level raises it by 2.3; the item's price level raises it by 1.1; the series age lowers it by 1.1.
- P90 order-up-to quantile: 37.8 units over the next 14 days (typical 51.0). The days-without-sales pattern lowers it by 15.6; the recent sales level raises it by 2.4; the same period last year raises it by 1.1.

**high velocity, P50 forecast near the segment's 90th percentile** (`FOODS_3_403_CA_1_evaluation`, origin 2015-01-04, velocity segment high):

- P50 forecast: 74.7 units over the next 14 days (typical 41.8). The recent sales level raises it by 17.2; the days-without-sales pattern raises it by 13.7; the item's price level raises it by 1.1.
- P90 order-up-to quantile: 109.8 units over the next 14 days (typical 79.1). The recent sales level raises it by 21.3; the same period last year raises it by 6.3; the item's price level raises it by 2.6.

**mid velocity, P50 forecast near the segment's 50th percentile** (`FOODS_3_724_CA_2_evaluation`, origin 2014-12-21, velocity segment mid):

- P50 forecast: 9.2 units over the next 14 days (typical 8.1). The days-without-sales pattern raises it by 0.9; the recent sales level raises it by 0.7; the same period last year raises it by 0.4.
- P90 order-up-to quantile: 16.0 units over the next 14 days (typical 15.3). The same period last year raises it by 2.0; the days-without-sales pattern lowers it by 1.3; the series age raises it by 0.3.

**mid velocity, P50 forecast near the segment's 90th percentile** (`FOODS_1_180_CA_1_evaluation`, origin 2015-01-18, velocity segment mid):

- P50 forecast: 18.2 units over the next 14 days (typical 22.8). The recent sales level lowers it by 4.1; the days-without-sales pattern raises it by 1.7; the series age lowers it by 0.7.
- P90 order-up-to quantile: 30.3 units over the next 14 days (typical 43.1). The days-without-sales pattern lowers it by 6.9; the recent sales level lowers it by 4.3; the calendar lowers it by 0.7.

**low velocity, P50 forecast near the segment's 50th percentile** (`HOBBIES_1_092_CA_2_evaluation`, origin 2014-12-14, velocity segment low):

- P50 forecast: 3.1 units over the next 14 days (typical 3.1). The days-without-sales pattern raises it by 0.4; the same period last year lowers it by 0.3; the calendar lowers it by 0.1.
- P90 order-up-to quantile: 7.2 units over the next 14 days (typical 5.9). The days-without-sales pattern raises it by 1.6; the same period last year lowers it by 0.6; the series age raises it by 0.4.

**low velocity, P50 forecast near the segment's 90th percentile** (`FOODS_1_159_CA_2_evaluation`, origin 2015-02-08, velocity segment low):

- P50 forecast: 7.1 units over the next 14 days (typical 7.2). The days-without-sales pattern raises it by 0.3; the same period last year lowers it by 0.2; the recent sales level lowers it by 0.2.
- P90 order-up-to quantile: 10.7 units over the next 14 days (typical 13.6). The recent sales level lowers it by 1.1; the days-without-sales pattern lowers it by 1.0; the same period last year lowers it by 0.3.

**largest price-vs-usual contribution to the service-level quantile** (`FOODS_3_446_CA_3_evaluation`, origin 2014-12-14, velocity segment high):

- P50 forecast: 56.2 units over the next 14 days (typical 100.4). The recent sales level lowers it by 30.8; the series age lowers it by 14.8; the same period last year raises it by 7.8.
- P90 order-up-to quantile: 182.4 units over the next 14 days (typical 189.8). The days-without-sales pattern lowers it by 70.4; the same period last year raises it by 64.8; the recent sales level lowers it by 53.2.

**a row where XGBoost's internal sort changed the quantile order** (`FOODS_2_101_CA_2_evaluation`, origin 2014-11-23, velocity segment low):

- P50 forecast: 0.0 units over the next 14 days (typical 1.8). The days-without-sales pattern lowers it by 0.9; the recent sales level lowers it by 0.5; the price (price is about usual) lowers it by 0.1.
- P90 order-up-to quantile: 1.2 units over the next 14 days (typical 2.7). The recent sales level lowers it by 0.9; the series age raises it by 0.8; the item's price level lowers it by 0.5.
