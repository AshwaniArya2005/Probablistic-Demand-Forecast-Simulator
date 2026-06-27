# Methodology

Everything here is decided in [`design.md`](design.md) (pre-registered rules, decision log) and reported in [`results/`](results/). This page is the readable summary.

## Data and target
- **Data:** the M5 (Walmart) sales, prices and calendar for stores CA_1 to CA_3, 2011 to 2016. 100 items drawn with a fixed seed, stratified by department and sales velocity, three stores each: **300 series**. Velocity segments (low / mid / high) are terciles fixed on data up to the tuning cutoff.
- **Closed days:** 25 December is a closed day in every year; its near-zero sales are treated as not being demand in rolling statistics (they stay in the actual sales that are replayed).
- **Target:** the cumulative demand over the next **P days from a Sunday review date**, P in {7, 10, 14}, forecast as a set of quantiles (0.10, 0.50, 0.80, 0.90, 0.95, 0.99). A P-day total is what an order-up-to policy needs: protection interval = review period 7 + lead time L, so P = 10 for L = 3 and 14 for L = 7.

## Evaluation design (the part that protects against fooling myself)
- **Tuning folds F1 to F4:** 12 consecutive Sunday origins each, all ending before 2015-08-30; F2 is the holiday fold. All model choices were made here.
- **Test window 2015-11-23 to 2016-05-22:** not used for any choice. Model versions v0 to v4 are refit every 8 weeks; each version's fit set ends 84 days before its cutoff (an embargo), and a 12-week calibration window follows the fit set. Tests assert the embargo and the schedule.
- **Pre-registration:** the rules for each phase (metrics, decision rules, thresholds, expected outcomes) were committed to `design.md` before any score was computed; a **touch log** records every use of test-window data, written before the run.
- **Leakage:** each feature has a written leakage note; tests perturb future sales and prices and check that no input changes.

## Models
- Baselines: naive, 28-day moving average (MA-28), seasonal naive. Point models: Linear Regression, Random Forest, XGBoost (squared error).
- **Quantile model:** one XGBoost multi-quantile model per horizon, trained on the **normalised target** y / (max(mean_28, floor) x P) (floor 1/7, chosen on the tuning folds), predictions scaled back, clipped at 0 and sorted. 35 inputs (calendar, price relative to usual, lags and rolling statistics, year-ago sum, zero-sales stretches; no series ids).
- **Benchmarks for the quantile forecast:** B1 (MA-28 + normal quantile with a pooled sigma), **B2** (XGBoost mean + normal quantile: the pre-specified "textbook" policy), and the **post-hoc B3a / B3c** (a point forecast plus empirical residual quantiles from the calibration window; defined after the tuning results were known, labelled as such, and used as the fair benchmark because they remove the normal-sigma assumption).
- Conformal calibration was pre-registered, tested and adds nothing (dropped).

## Metrics
- **Scaled pinball loss** over the six quantiles, each series scaled by its naive-P error before the version's cutoff (lower is better); mean and median over series.
- **Coverage of the ordered quantity:** share of demand at or below ceil(q), compared with the nominal level with a discrete-aware rule (tolerance 0.03), because demand is integer and ties are large for slow movers.
- **WAPE** for point and median forecasts; MASE as a reference.
- **Uncertainty:** cluster bootstrap over the 100 items (an item's three store-series resampled together), 10,000 resamples, seed 0. It reflects which items were sampled, not variation between periods.

## Inventory replay (descriptive)
- Periodic review every Sunday, order-up-to level S (integer, `ceil`), orders arrive before opening of day t + L + 1, lost sales, demand replayed from actual M5 sales. Simulation starts four weeks before the first counted review (warm-up excluded).
- **Policies:** naive `ceil(c x mean_28 x P)` over a grid of c (widened by a pre-registered rule, then frozen); the point policy (B2, with a labelled sqrt-scale variant where the constant-variance check fails); the quantile policy `ceil(q_alpha)`; and the post-hoc B3a. alpha in {0.80, 0.90, 0.95, 0.99}.
- **Metrics:** fill rate (units served / demanded), cycle service (share of cycles without a stockout), average on-hand inventory; costs are a sensitivity sweep, not a headline.
- **Matched-service statistic:** the inventory one policy needs to reach the other's fill rate, by linear interpolation, no extrapolation, reported in two anchorings (at the quantile policy's own settings, and at the comparator's own settings). Because each curve has four points the interpolation is coarse.
- **What it means:** a **shuffled-demand null test** shows the statistic is positive even when demand is unrelated to the forecasts, so it is a description of two rules under real demand, not evidence of forecast skill.
- **Sensitivities (each pre-registered, run once):** lead time 7, random lead time, no future-price inputs, a hindsight zero-demand-cycle diagnostic; plus a post-hoc extension of the naive grid for the cost sweep.

## Checks on the checker
Independent hand recomputation of one series over ten review weeks (tuning and test window: all agree); a shuffled-demand null test (tuning and test window); a random-lead-time follow-up (draws applied, not saturation); a rest-period diagnostic; a clean-clone reproduction of the data pipeline and tests
(which found and fixed one defect); a retrain of one frozen version from scratch compared within a stated tolerance; recorded hashes of every frozen model file; a stockout-risk ROC/PR diagnostic on the tuning folds (exploratory, no retraining) that independently corroborates the Phase 9 label
investigation — discrimination is weak (ROC AUC 0.59) at the reference level where near-zero medians and outages dominate, and improves (AUC 0.83) further into the tail (`docs/results/phase11_stockout_roc_tuning.md`, curves in `docs/figures/stockout_roc_curves.png` / `stockout_pr_curves.png`).

## Figures
Pictorial versions of the tables above, generated by `ml/metrics_figures.py` from the same committed result files (no new numbers): test-window scaled pinball by method and horizon, achieved-vs-nominal coverage, and the stockout ROC/PR curves — all in `docs/figures/`, alongside the three Phase 3 EDA plots from `ml/eda.py`.

## Explanations
Exact per-quantile contributions from single-target boosters sliced from the joint model (XGBoost 3.4.1's joint `pred_contribs` does not reconcile with `predict`), converted to units and grouped into themes for planner sentences ("price is X% below usual", never "promotion"). Associations, not causal effects. Tuning-fold rows only until a final frozen explanation run.
