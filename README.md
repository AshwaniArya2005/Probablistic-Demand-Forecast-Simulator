# Probabilistic Demand Forecasting & Inventory Policy Simulator

Forecasts the **distribution** of demand over a protection interval (7, 10 or 14 days) for retail item-store series, and is designed to compare
inventory ordering policies that use those forecasts in a historical replay. Built on the M5 (Walmart) data, 300 California series.

**Status: work in progress.** Data preparation, features, baselines, point and quantile models and their evaluation on the tuning folds are done
(phases 2 to 7); explanations (phase 8) are being built. **The inventory simulator, API and dashboard do not exist yet, and no result below is a
claim about inventory or cost.** Every number in this file comes from a logged run; the decision log and the pre-registered rules are in
[`docs/design.md`](docs/design.md), the run outputs in [`docs/results/`](docs/results/).

## What was evaluated, and how

- **Target:** cumulative demand over the next P days from a Sunday review date, for P in {7, 10, 14}, as a full quantile forecast (0.10, 0.50, 0.80, 0.90, 0.95, 0.99).
- **Data:** stores CA_1 to CA_3, 100 items sampled with a fixed seed and stratified by department and sales velocity: 300 series, 2011 to 2016.
- **Models:** naive, MA-28 and seasonal-naive baselines; Linear Regression, Random Forest and XGBoost for point forecasts; XGBoost quantile regression; optional conformal calibration.
- **Honest evaluation:** all model choices were made on four rolling-origin *tuning folds* that end before 2015-08-30, with rules committed to `docs/design.md` **before** scoring. The **test window (2015-11-23 to 2016-05-22) has not been evaluated**; a touch log records every use of it (currently empty).
- **Tests:** a fast synthetic suite (run by CI) plus real-data checks; a phase is done only when its tests pass and are recorded ([`docs/testing.md`](docs/testing.md)).

## Results so far (tuning folds only)

**Point forecasts** ([phase 6](docs/results/phase6_point_tuning.md), [bootstrap](docs/results/phase6_point_bootstrap.md)). Against the MA-28 baseline, pooled WAPE ratio at P = 7 / 10 / 14: Random Forest 0.962 / 0.957 / 0.958 (better in all 12 fold-by-horizon cells; difference -0.0157, 95% interval [-0.0228, -0.0085] over items), XGBoost 0.985 / 0.967 / 0.978 (11 of 12 cells; -0.0091 [-0.0164, -0.0010]), Linear Regression about 1.00 (no difference). The gain comes from mid- and high-velocity series: for low-velocity series no learned point model beat MA-28.

**Quantile forecasts** ([phase 7](docs/results/phase7_quantile_tuning.md), [cells won and bootstrap](docs/results/phase7_bootstrap.md)). Mean scaled pinball loss (lower is better):

| Method | Mean scaled pinball |
|---|---|
| XGBoost quantile model | 0.2639 |
| Pre-specified point policy (XGBoost mean + normal quantile, pooled RMSE sigma) | 0.4128 |
| Post-hoc: same point forecast + empirical residual quantiles from the calibration window | 0.3025 |

The quantile model wins in all 12 cells against every one of six benchmarks, and every bootstrap interval excludes zero. Its coverage of the ordered quantity is consistent with the nominal levels 0.80, 0.90, 0.95 and 0.99 in all 12 cells; conformal calibration adds nothing on these folds. The normalised target beats the raw target by about 3% once both are trained comparably.

**Where the advantage is.** Against the pre-specified point policy the quantile model's advantage **concentrates in high-velocity series** (scaled pinball difference -0.340 for high, -0.071 for mid, -0.036 for low velocity). That concentration is mostly a property of the normal-quantile point policy, not of the quantile model: against the post-hoc empirical-residual-quantile policy the advantage is **small (-0.039, 12.8% of that policy's loss) and roughly even across segments** (-0.046, -0.035, -0.034). About three quarters of the headline gap to the pre-specified policy disappears once the normal-sigma assumption is replaced. The post-hoc benchmark was defined after the results were known and is labelled as such.

## Limitations

1. No inventory levels, lead times or costs exist in the data; all are simulated assumptions (not yet simulated).
2. **Sales are not demand.** History is censored when stock ran out. About 11% of high-velocity days (15% mid, 34% low) sit inside zero runs of 14 or more days that look like stock-outs, delistings or seasonal absence. The data is not filtered for this and the forecasts are of sales, a proxy for demand.
3. There is no promotion column. Price is compared with the item's usual price, and explanations say "price is X% below usual", never "promotion". Model attributions are associations, not causal effects.
4. Anonymised items, one retailer, 2011 to 2016; 300 series from three stores.
5. Future prices are treated as known at order time (an assumption); removing them changes the quantile model's loss by about 0.2%.
6. The four tuning folds were used both to select and to report, so fold scores are mildly optimistic; the item bootstrap captures which items were sampled, not variation between periods. The scale floor for target normalisation sits at the top edge of its candidate range.

## Reproduce

Python 3.12 with [uv](https://docs.astral.sh/uv/). The M5 data is not included (download it from Kaggle under its own rules and put `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv` in `data/raw/`).

```
uv sync
uv run pytest                    # fast suite, no data needed
uv run python ml/prepare.py      # clean the data, derive the closed-day rule and velocity segments, draw the 100-item subset
uv run python ml/features.py     # features and targets
uv run pytest -m realdata        # real-data checks
```
