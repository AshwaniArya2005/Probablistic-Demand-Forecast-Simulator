# Probabilistic Demand Forecasting & Inventory Policy Simulator

Forecasts the **distribution** of demand over a protection interval (7, 10 or 14 days) for retail item-store series, and is designed to compare
inventory ordering policies that use those forecasts in a historical replay. Built on the M5 (Walmart) data, 300 California series.

**Status: work in progress.** Data preparation, features, baselines, point and quantile models, their evaluation on the tuning folds, the explanation
layer and the planner-facing quantities (reorder point, safety stock, stockout-risk label) are done (phases 2 to 9). The base-case inventory replay has been run once on the test window (phase 10); its sensitivities, the API and the dashboard do not exist yet.
**Inventory results are historical replays under stated assumptions (no real stock, lead-time or cost data exist), not claims about a real retailer.** Every number in this file comes
from a logged run; the decision log and the pre-registered rules are in [`docs/design.md`](docs/design.md), the run outputs in [`docs/results/`](docs/results/).

## What was evaluated, and how

- **Target:** cumulative demand over the next P days from a Sunday review date, for P in {7, 10, 14}, as a full quantile forecast (0.10, 0.50, 0.80, 0.90, 0.95, 0.99).
- **Data:** stores CA_1 to CA_3, 100 items sampled with a fixed seed and stratified by department and sales velocity: 300 series, 2011 to 2016.
- **Models:** naive, MA-28 and seasonal-naive baselines; Linear Regression, Random Forest and XGBoost for point forecasts; XGBoost quantile regression; optional conformal calibration.
- **Honest evaluation:** all model choices were made on four rolling-origin *tuning folds* that end before 2015-08-30, with rules committed to `docs/design.md` **before** scoring. The **test window (2015-11-23 to 2016-05-22) has been used once, for the base-case inventory replay** (its touch-log entry was written before the run and the code was frozen first); every use is recorded in the touch log.
- **Tests:** a fast synthetic suite (run by CI) plus real-data checks; a phase is done only when its tests pass and are recorded ([`docs/testing.md`](docs/testing.md)).

## Results so far (tuning folds only)

**Headline.** Against the post-hoc benchmark that keeps the same point forecast but takes its quantiles from empirical residuals, the quantile model has **about 13% lower scaled pinball on the tuning folds** (0.2639 vs 0.3025, better in 12 of 12 fold-by-horizon cells, difference -0.039, 95% interval [-0.049, -0.031] over items). **The normal sigma was most of the gap to the textbook policy** (a normal quantile around the mean forecast with a pooled RMSE sigma, 0.4128): replacing it removes about three quarters of that gap. The gap to the textbook policy is never quoted without this.

**Point forecasts** ([phase 6](docs/results/phase6_point_tuning.md), [bootstrap](docs/results/phase6_point_bootstrap.md)). Against the MA-28 baseline, pooled WAPE ratio at P = 7 / 10 / 14: Random Forest 0.962 / 0.957 / 0.958 (better in all 12 fold-by-horizon cells; difference -0.0157, 95% interval [-0.0228, -0.0085] over items), XGBoost 0.985 / 0.967 / 0.978 (11 of 12 cells; -0.0091 [-0.0164, -0.0010]), Linear Regression about 1.00 (no difference). The gain comes from mid- and high-velocity series: for low-velocity series no learned point model beat MA-28.

**Model performance, quantile forecasts** ([phase 7](docs/results/phase7_quantile_tuning.md), [cells won and bootstrap](docs/results/phase7_bootstrap.md)):

| | Value | Note |
|---|---|---|
| Mean scaled pinball, XGBoost quantile model | 0.2639 | median over series 0.1861 |
| Mean scaled pinball, post-hoc B3a (point forecast + empirical residual quantiles) | 0.3025 | the quantile model is about 13% lower, 12 of 12 cells |
| Mean scaled pinball, textbook point policy (normal sigma) | 0.4128 | its normal quantiles over-cover the 0.80 and 0.90 levels in all 12 cells |
| Coverage of the ordered quantity (0.80 / 0.90 / 0.95 / 0.99) | consistent with nominal in 12 of 12 cells at every level | the test is discrete-aware and weak where ties are large |
| Quantile crossing before sorting (XGBoost sorts internally) | **2.4% of rows on average** (0.9% to 4.7% by cell) | corrected: an earlier report of 0.000 measured already-sorted output |
| Conformal calibration | adds nothing | 0.2659 vs 0.2639; enough scores in every segment |
| Normalised vs raw target | about 3% better once both are trained comparably | 7% in the pre-registered comparison; the round cap penalised the raw fit |

**Where the advantage is.** Against the textbook point policy the quantile model's advantage **concentrates in high-velocity series** (scaled pinball difference -0.340 for high, -0.071 for mid, -0.036 for low velocity). That concentration is mostly a property of the normal-quantile policy: against the post-hoc empirical-residual policy the advantage is **small and roughly even across segments** (-0.046, -0.035, -0.034). The post-hoc benchmark was defined after the results were known and is labelled as such.

**Explanations** ([phase 8](docs/results/phase8_explanations_tuning.md)). Exact per-quantile contributions are converted to units and grouped into themes for planner sentences ("price is X% below usual", never "promotion"). Two facts a reader needs:
- XGBoost 3.4.1's joint `pred_contribs` for a multi-quantile model, and the `shap` package that wraps it, **do not reconcile with `predict`** (they explain the unsorted per-target tree sums while `predict` returns them sorted), so contributions here come from single-target boosters sliced from the model and are additive to float precision.
- On **89 of 3,600 explained rows (2.5%)** in the explained fold, the quantile is explained through **another quantile's trees**, because the sort changed the order.

**Inventory replay on the test window** ([phase 10 primary run](docs/results/phase10_simulator_primary.md), base case: lead time 3 days, weekly Sunday review, lost sales, 25 review dates, one run made once). The **fair benchmark is B3a**, the same point forecast with empirical residual quantiles. **Against B3a the quantile policy needs about 5% to 16% less inventory at the same fill rate for service targets 0.80 to 0.95** (depending on how the coarse four-point curves are interpolated; item-bootstrap intervals from about 0 to 25%, and they include zero in the low- and high-velocity segments and in the holiday peak), **and much less at 0.99** (37.7 against 69.7 units per series-day for the same fill rate). Against the pre-specified normal-sigma point policy (B2) the reduction is about 29%; against the naive c x average-demand rule about 45%. The intervals reflect which items were sampled, not variation between periods (one 26-week test period). Three predictions were committed before the run: **0.99 is inefficient for the quantile policy (confirmed); a smaller edge over B3a than over the point policy (holds pooled and for high velocity, fails for mid velocity, so not confirmed); under-service in the rest period (refuted: the quantile policy served above its target at every level in both sub-periods).** At the cost ratio for a 0.80 target the naive rule had the lowest total cost, at the edge of its grid. The point policy's normal-sigma assumption failed the constant-CV check for high-velocity series, so its deficit there partly measures that assumption.

**Planner quantities** ([phase 9](docs/results/phase9_policy_quantities_tuning.md)). Reorder points and safety stock from the quantile model are close to nominal coverage at 0.80 to 0.95 (0.844 / 0.917 / 0.960), with mean safety stock 4.1 / 7.5 / 11.3 units; the classic normal-sigma version over-protects at every level below 0.99 (coverage 0.944 / 0.969 / 0.979; safety stock 28.2 / 42.7 / 54.7), mostly in high-velocity series. The constant-CV assumption behind the classic sigma **fails the pre-registered check** for high-velocity series (all 12 cells) and half the mid-velocity cells, so the classic figures are not a general statement about textbook policies. The stockout-risk label (HIGH / MEDIUM / LOW plus an overstock flag) orders realised shortfall frequency correctly, but it is a **risk band, not a probability**. Investigated ([label investigation](docs/results/phase9_label_investigation_tuning.md)): no boundary bug, and the median itself is well calibrated. HIGH rows ran short less often than the label suggests where the median is below one unit (rounded up to 1: shortfall 0.17) and inside a long zero run (an outage whose end the model cannot know); where the median exceeds 2 units, HIGH rows ran short 54% to 68% of the time. Any display of the label carries those two qualifiers. No cost or service result exists yet.

Planner sentences do not interpret the series-age input (it sits in a neutral "remaining factors" theme); attributions are associations, not causal effects.

## Known limitation and future work

A **post-hoc, report-only** ablation dropping the series-age input (`age_days`) lowered tuning-fold scaled pinball from 0.2639 to 0.2594 (-1.7%). It was not pre-registered and the model is unchanged: every result here, including the simulator, uses the frozen feature set. Dropping it is future work and would be a new pre-registered decision with its own evidence.

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
