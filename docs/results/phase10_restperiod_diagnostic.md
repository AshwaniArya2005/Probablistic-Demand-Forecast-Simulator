# Rest-period service: diagnostic (test window, base case, quantile policy)

Run 2026-09-21. Peak = cycles starting on or before 2016-01-03 (6 reviews), rest = the other 19. `coverage_of_ceil_q` = share of review origins whose realised P-day demand was at or below the
order-up-to level ceil(q); `cycle_service` = share of series-cycles without a stockout in the simulation; `wedge` = their difference; `positive_demand_only` drops series-cycles whose protection-interval demand was zero.

|   alpha | period   |   review_dates |   coverage_of_ceil_q |   cycle_service |   wedge |   cycle_service_positive_demand_only |   zero_demand_share |
|--------:|:---------|---------------:|---------------------:|----------------:|--------:|-------------------------------------:|--------------------:|
|    0.8  | peak     |              6 |               0.8544 |          0.8817 |  0.0272 |                               0.8589 |              0.1611 |
|    0.8  | rest     |             19 |               0.8498 |          0.8858 |  0.036  |                               0.8699 |              0.1225 |
|    0.8  | all      |             25 |               0.8509 |          0.8848 |  0.0339 |                               0.8673 |              0.1317 |
|    0.9  | peak     |              6 |               0.9183 |          0.9322 |  0.0139 |                               0.9192 |              0.1611 |
|    0.9  | rest     |             19 |               0.9268 |          0.9432 |  0.0163 |                               0.9352 |              0.1225 |
|    0.9  | all      |             25 |               0.9248 |          0.9405 |  0.0157 |                               0.9315 |              0.1317 |
|    0.95 | peak     |              6 |               0.9589 |          0.9667 |  0.0078 |                               0.9603 |              0.1611 |
|    0.95 | rest     |             19 |               0.9616 |          0.9719 |  0.0104 |                               0.968  |              0.1225 |
|    0.95 | all      |             25 |               0.9609 |          0.9707 |  0.0097 |                               0.9662 |              0.1317 |
|    0.99 | peak     |              6 |               0.9906 |          0.995  |  0.0044 |                               0.994  |              0.1611 |
|    0.99 | rest     |             19 |               0.9937 |          0.9977 |  0.004  |                               0.9974 |              0.1225 |
|    0.99 | all      |             25 |               0.9929 |          0.9971 |  0.0041 |                               0.9966 |              0.1317 |

## Realised demand against the forecasts

| period   |   sum y / sum yhat (XGBoost mean) |   sum y / sum MA-28 |   mean P-day demand per series |   zero-demand share of series-origins |
|:---------|----------------------------------:|--------------------:|-------------------------------:|--------------------------------------:|
| peak     |                            0.958  |              0.9621 |                         13.008 |                                0.1611 |
| rest     |                            0.9624 |              0.9808 |                         13.475 |                                0.1225 |
| all      |                            0.9614 |              0.9764 |                         13.363 |                                0.1317 |
