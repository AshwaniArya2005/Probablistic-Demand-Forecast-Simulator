# Stockout-risk ROC/PR diagnostic (tuning folds, exploratory; design.md 2026-09-22 pre-registration)

Run 2026-09-22. Ground truth, reference positions and the interpolated score are fixed by the pre-registration (not tuned here). No retraining; the frozen quantile model's own six quantiles
(q10, q50, q80, q90, q95, q99) are the only inputs to the score. Reported as computed, including any weak slice.

|   c | slice   |     n |   positive_rate |   roc_auc |   average_precision |
|----:|:--------|------:|----------------:|----------:|--------------------:|
| 1   | pooled  | 28800 |          0.3414 |    0.5883 |              0.3752 |
| 1   | P=10    | 14400 |          0.3219 |    0.582  |              0.3502 |
| 1   | P=14    | 14400 |          0.3609 |    0.5921 |              0.3983 |
| 1.5 | pooled  | 28800 |          0.1375 |    0.7271 |              0.2417 |
| 1.5 | P=10    | 14400 |          0.1281 |    0.7123 |              0.2165 |
| 1.5 | P=14    | 14400 |          0.1468 |    0.7399 |              0.2664 |
| 2   | pooled  | 28800 |          0.0741 |    0.8251 |              0.203  |
| 2   | P=10    | 14400 |          0.0725 |    0.8168 |              0.1874 |
| 2   | P=14    | 14400 |          0.0758 |    0.833  |              0.2198 |

Full-resolution ROC and precision-recall curve points (downsampled to at most 40 points per curve for size): `docs/results/phase11_stockout_roc_curves.csv`.

Reading: this is a diagnostic of how well the model's existing quantile grid discriminates shortfall when read as a continuous score; it does not change the stockout-risk label's cut points
(P50 / P90 / P99, design.md Phase 9 pre-registration) or any policy, and a low AUC in a slice is reported as such, not adjusted for.
