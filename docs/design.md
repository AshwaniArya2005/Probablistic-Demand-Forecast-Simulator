# Design: Probabilistic Demand Forecasting & Inventory Policy Simulator

Status: approved with amendments, 2026-09-20. This is the source of truth for Phases 2-14.
Where a later phase needs to change something here, edit this file and add a line to the decision log (section 12).

Repo name: `probabilistic-demand-forecasting-simulator` (the folder was originally called "Supply Chain Churn Prediction", a misnomer: this is demand forecasting, not churn).

## 1. What we will and will not claim

- Every number in the README and on the resume must come from a logged run or simulation output. Nothing is quoted before it is measured.
- The headline result is an **inventory-vs-service-level trade-off curve at matched service levels** (section 9), not a single cost number.
- If the quantile policy does not win somewhere (e.g. fast movers, or at 0.99), that is reported as is.
- Simulation results describe *historical replay under stated assumptions*. They are not a claim about real savings.

## 2. Dataset and subset

**M5 Forecasting (Walmart), from Kaggle.** Download manually (account + accepting competition rules). Check the licence on the competition Data page. Raw files live in `data/raw/` and are git-ignored.

| File | Contents (from the competition docs; verified on load in Phase 2) |
|---|---|
| `sales_train_evaluation.csv` | wide format, one row per item-store series; id columns (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`) and daily unit sales `d_1..d_1941` |
| `calendar.csv` | date, weekday, month, year, `wm_yr_wk` (price join key), `event_name/type_1/2`, `snap_CA/TX/WI` |
| `sell_prices.csv` | weekly `sell_price` per store and item, keyed by `wm_yr_wk` |

Data spans 2011-01-29 to 2016-05-22.

**Subset:** state CA, stores CA_1..CA_3, 100 items sampled with a fixed seed, stratified by department and by sales-velocity tercile so slow and intermittent items are represented. That is 300 series (about 580k daily rows). Items without at least 2 years of history before the test window are dropped and the count is documented. Scale up only if runtime allows.

**Velocity segments (high / mid / low):** terciles of each series' mean daily sales since its first on-sale week, computed **once, from data up to the tuning cutoff (`test_start - 12 weeks`, section 5) only**. Items are sampled stratified on their mean velocity tercile across the three stores; each series carries its own segment label. The same labels drive the stratified sampling, the pooling of sigma and conformal offsets (section 5), and the result breakdowns (section 9). They are never recomputed with later data, because a segment defined using test-period sales would leak the outcome into how results are grouped and calibrated.

**Cleaning rules known in advance:** rows before an item's first `sell_price` are "not yet on sale", not zero demand, and are dropped. Days when the store is closed (e.g. Christmas, to be verified in EDA) are structural zeros and are flagged.

**Limitations (must appear in the README):**
1. No inventory levels, lead times, or costs in the data. All are simulated assumptions.
2. Sales are not demand: history is censored when stock ran out. We forecast sales as a proxy for demand.
3. No promotion column. We derive `price vs. usual` as a proxy. Model outputs and SHAP text say "price is X% below usual", never "promotion".
4. Anonymised items, one retailer, 2011-2016.

## 3. Target, horizon, quantiles

**Target:** cumulative demand over the next P days from forecast origin t, using only information available at the end of day t:

`Y(s, t, P) = sales[s, t+1] + ... + sales[s, t+P]`

Why cumulative: daily SKU-store demand is mostly 0-2 units, so a daily interval is meaningless, and quantiles of daily forecasts cannot be summed to get the quantile of lead-time demand (the sum of P90s equals the P90 of the sum only under perfect dependence, so it over-orders). Training directly on the P-day sum gives the right quantity, with day-to-day dependence handled implicitly and no distributional assumption. Cost: one model per horizon, and overlapping windows make consecutive training rows correlated (evaluation uses weekly origins for that reason). Rejected alternatives: Monte Carlo from daily distributions (needs a copula), parametric negative binomial (extra assumption; possible stretch).

**Horizons:** review period R = 7 days, lead time L in {3, 7}, protection interval P = R + L in {10, 14}, plus P = 7 for the "7-day forecast" display. So **P in {7, 10, 14}**.

**Quantile set:** {0.10, 0.50, 0.80, 0.90, 0.95, 0.99}.
- P10 / P50 / P90 give the displayed forecast and the 80% interval.
- {0.80, 0.90, 0.95, 0.99} are the service-level quantiles used by the policy.
- Sorted (monotone rearrangement) after prediction to fix quantile crossing.
- 0.99 is a tail estimate on limited data. It is evaluated explicitly (one-sided coverage), not assumed to work.

**Global model:** one model per horizon across all 300 series, with item/store/department as features.

**Target normalization (experiment, decided on validation):** train on `Y / scale(s, t)` with scale = trailing mean daily sales (past data only, floored) times P, then multiply predictions back. Valid because quantiles commute with multiplication by a positive constant. It makes training loss equal to scaled pinball (each series weighted equally, not dominated by fast movers). Risks: floor choice for near-dead series, lumpiness at small scales. A/B against the raw target on the tuning folds using the scaled metrics of section 6; the choice is frozen before the test window is touched.

## 4. Features (as of origin t; details and leakage notes in Phase 4)

- **History (all computed from data <= t):** recent daily sales lags, rolling mean/std over 7/14/28/91 days, fraction of zero days, days since last sale, sum of the last P days, sum of the same window 364 days earlier, current price, item/store/dept/cat ids.
- **Future-window covariate aggregates (window t+1..t+P):** number of SNAP days (state-specific), number of event days (by type where useful), number of weekend days, number of store-closed days (a calendar rule, see section 5), **planned price relative to trailing median** (mean and min window price divided by the trailing median price, the median computed from prices <= t).
- **Known-prices assumption (stated, not hidden):** future prices, events and SNAP days are treated as known at order time. Events and SNAP schedules genuinely are. Prices are an assumption: M5 gives us realised prices, and we treat them as if they were the plan. That flatters ML policies relative to the naive rule, so we run an ablation without future-price features and report the difference.
- Future sales are never a feature. Predictive effects of price are associations, not causal estimates (why "what-if" price sliders are out of scope).

## 5. Splits and leakage rules

Timeline: **test = last 26 weeks** (about Nov 2015 - May 2016). Model selection and tuning use rolling-origin folds on data up to `test_start - 12 weeks` only.

For a model version with cutoff date `c` (retrained every K = 8 weeks on an expanding window):
- **Fit set:** origins `s` with `s + P <= c - 84 days`. The target window must have fully ended before the cutoff (embargo), otherwise labels leak across the train/test boundary.
- **Calibration window:** the 12 weeks before `c`. Out-of-sample residuals from here give the point policy's sigma and the optional conformal offsets. Same window, same pooling (below) for both policies.
- **Use:** review dates in `[c, c + K weeks)`.
- **Model staleness is 12 to 20 weeks, not 12.** The fit set ends at `c - 84 days` and a version is used for review dates up to `c + 8 weeks`, so the newest data the model has seen is 12 weeks old at the start of its use period and 20 weeks old at the end. Simplification (`ponytail`): the model is not refit on the calibration window. Upgrade path: refit after calibrating (staleness drops to 0-8 weeks), keeping the calibration offsets, at the price of slightly optimistic residuals.
- **Calibration is pooled by velocity segment, never estimated per series alone, for both policies.** A 12-week window gives about 12 weekly origins per series (about 84 daily origins, but overlapping and strongly dependent). That is too few for a reliable per-series sigma, and it cannot support a 0.99 conformal offset at all (a 0.99 quantile needs on the order of 100 scores; the maximum of 12 is not that). Rule: residuals and conformal scores are divided by the series' scale (P x trailing mean daily sales, floored; the same scale as the normalization in section 3), pooled within the velocity segment, and the pooled statistic is multiplied back by the series' own scale: `sigma_s = sigma_segment * scale_s`, `offset_s,alpha = offset_segment,alpha * scale_s`. This assumes a roughly constant coefficient of variation within a segment; Phase 9 checks it with a residual-vs-scale diagnostic (fallback: scale by the square root of the scale, which is Poisson-like). Shrinking each series toward its segment value is the alternative if pooling is too coarse. Same window, same segments, same rule for the point policy's sigma and the quantile policy's offsets, so neither gets better-calibrated inputs than the other.

General rules: never shuffle; nothing (scalers, price medians, scale factors, item sampling thresholds) is fitted on data after the origin; all history features are computed as-of t; test data is never used for tuning; the test-window forecasts are computed once and reused across policies (forecasts depend only on historical sales, not on simulated stock).

**Origins:** an origin is evaluated only if its whole target window lies inside the data, so the last usable origin for horizon P is `2016-05-22 - P days` and the effective test window is slightly under 26 weeks for the longer horizons.

**Test window regime (read before interpreting any result).** The last 26 weeks (about 22 Nov 2015 to 22 May 2016) include the Thanksgiving-Christmas-New Year peak, **closed-store days** such as Christmas Day (structural zeros; to be verified in EDA), the Super Bowl and Easter.
- Results mix a holiday-peak regime with a normal one. We therefore report two sub-periods with boundaries fixed now: **holiday peak = 22 Nov 2015 to 3 Jan 2016**, and **rest = after 3 Jan 2016**. We do not claim results generalise to other seasons.
- Closed days are known from the calendar, so the future-window closed-day count is a legitimate feature. It is identified as a calendar rule (e.g. Dec 25) from training years only, never from zeros inside the window being forecast. In the simulator, closed days replay zero demand while holding cost still accrues.
- Calibration windows lag the regime. Version 1's calibration window (autumn, calm) is applied through the holiday peak, so sigma and conformal offsets will likely be too small there; the next version's window contains the peak and will be too large afterwards. Expect achieved-vs-target service gaps in both policies; conformal prediction carries no exchangeability guarantee for time series.
- The first model versions were fit on data ending around late August 2015, so they have seen only four earlier holiday seasons (2011-2014).

## 6. Metrics (scaled, so series of different volume are comparable)

- **Point:** WAPE = sum|y - yhat| / sum y (pooled). **MASE** per series = test MAE / in-sample MAE of the naive-P forecast (`y_t - y_{t-P}`) on the training period, averaged over series. Series with zero scale are excluded and counted. Raw MAE/RMSE are reported for reference. MAPE is shown only on non-zero targets, to demonstrate that it breaks at zero, and is never used for selection.
- **Probabilistic:** **scaled pinball loss** = pinball loss divided by the same per-series scale as MASE, averaged over quantiles and series. Also **coverage**: empirical share of actuals inside [P10, P90] (nominal 0.80) and one-sided coverage of each quantile (share of `y <= q_alpha` vs alpha). One-sided coverage is what the reorder policy needs. Because demand is a small integer, nominal-or-higher coverage is expected for slow movers, and is explained rather than hidden.
- Evaluation origins are weekly.

## 7. Models

Baselines for the P-day sum: **naive** (previous P days), **MA-28** (28-day mean x P/28), **seasonal naive** (same window 364 days earlier, weekday-aligned). "Same weekday last week" collapses into naive for a 7-day sum, hence the swap. Then Linear Regression, Random Forest, XGBoost (squared error = mean forecast), XGBoost quantile (`reg:quantileerror`, LightGBM as fallback). ARIMA is dropped. Conformal calibration (split conformal on the calibration window, scores pooled by velocity segment as in section 5) is a cheap stretch, reported as coverage before vs after.

## 8. Inventory policies (all periodic-review, order-up-to level S)

Every policy sets S; order quantity = `max(0, S - inventory position)` (inventory position = on-hand + on-order). **S is rounded up with `ceil` to an integer for every policy** (same rule for all, so no policy benefits from the rounding; units are indivisible; a discrete quantile is the smallest integer whose CDF reaches the target).

Service-level grid for target-driven policies: **alpha in {0.80, 0.90, 0.95, 0.99}**.

- **Naive rule:** `S = ceil(c * mu28 * P)`, where `mu28` is mean daily sales over the 28 days ending at t and `c` is a cover multiplier. This is the "hold c times average demand over the protection interval" rule practitioners use. It has no service-level target, so it is swept over a grid of `c` (about 1.0 to 3.0, widened until its achieved service overlaps the other policies) to draw its own curve. `c = 1` is the plain "order up to average demand" point.
- **Point policy:** `S = ceil(max(0, yhat + z_alpha * sigma_s))`, where `yhat` is the squared-error XGBoost mean forecast for horizon P and `z_alpha` the normal quantile. `sigma_s = sigma_segment * scale_s`, where `sigma_segment` is the **RMSE of scale-normalised, out-of-sample residuals of the P-day forecast on the calibration window, pooled over all series in the velocity segment** (section 5); per-series estimates are too noisy. This is the textbook `z * sigma_daily * sqrt(P)`, with `sigma_daily * sqrt(P)` estimated directly at horizon P and pooled by segment. RMSE (not std) so any bias is charged to the policy. Assumes normal, independent demand: this is the assumption the quantile policy relaxes.
- **Quantile policy:** `S = ceil(q_alpha)`, the direct P-day quantile from the sorted quantile model (optionally conformally calibrated, with offsets pooled by segment as in section 5).
- Newsvendor link (for the README): ordering up to the `Cu/(Cu+Co)` quantile balances the cost of one unit too few against one too many. The service level is that critical ratio expressed as a probability.

## 9. Simulator

**Assumptions**

| Assumption | Choice | How it could be wrong |
|---|---|---|
| Demand | Actual M5 sales replayed | Understates true demand where the store was out of stock |
| Information | Planners see the true demand history even when simulated stock ran out (forecasts and the naive rule read historical M5 sales, not simulated sales) | Real systems only see censored sales; optimistic for all policies |
| Unmet demand | Lost sales | Backorder variant available as a sensitivity |
| Policy | Periodic review R = 7, order-up-to S, one fixed review weekday for all series | Real retailers stagger reviews by supplier |
| Lead time | Fixed L = 3 base (grid {3, 7}) | No supplier data; random lead time is a sensitivity |
| Timing | Order placed after close of day t, arrives before opening day t+L+1, so the protection interval is exactly days t+1..t+R+L, matching the target | |
| Start | Stock at S at the first review, 4-week warm-up excluded from metrics | |
| Prices/costs | Unit price = item-store median price in the test window; unit cost = 70% of price; holding = 25%/year of unit cost per unit-day; stockout = lost margin (30% of price). Fixed before any results. | Illustrative; not from data |

**Headline result: trade-off curves at matched service levels.** For each policy (naive over `c`, point and quantile over alpha), plot **average on-hand inventory (units) against achieved service**. Two panels: **fill rate** (units served / units demanded, pooled; the business-facing metric and the primary panel) and **cycle service level** (share of review cycles without a stockout; what the quantile target alpha nominally controls, so also a calibration check of the decisions). Comparison = inventory needed to reach the same achieved service, linearly interpolated between points, no extrapolation, with the coarseness of a four-point grid stated. 95% confidence intervals from a **cluster bootstrap over items**: an item's three store-series are resampled together because they share the item's demand drivers and are not independent, so the effective sample size is 100 items, not 300. The bootstrap does not capture period-to-period variability (there is one 26-week test period), so the intervals understate uncertainty about other seasons; the README says so. Results are also broken down by velocity segment (high / mid / low, defined in section 2) and by the two test sub-periods (section 5).

**Cost metric is a sensitivity sweep, not the headline.** With the default costs, the cost-implied critical ratio is: holding per unit over one 7-day cycle = 0.7 x 0.25/365 x 7 x price = about 0.0034 x price; stockout = 0.3 x price; ratio = about 89, critical ratio about 0.989 (about 99%). So at default costs a cost-optimiser would simply choose the top of the grid, and a cost "win" would say more about the cost assumption than the policy. Instead we sweep the stockout-to-holding cost ratio rho over {4, 9, 19, 99}, whose critical ratios are {0.80, 0.90, 0.95, 0.99}, and report each policy's minimum total cost over its own settings at each rho. (Newsvendor critical ratio is a textbook approximation for periodic-review lost sales; the simulation is the arbiter.)

**Other sensitivities:** lead time L in {3, 7}; random lead time L ~ Uniform{2, 3, 4} (same mean 3 as the nominal, isolates variance; policies still plan with nominal L; orders may cross); no-future-price ablation; optional backorder variant.

**Reporting:** stockout rate, fill rate, average inventory, total cost, plus achieved-vs-target service. Written up even if quantile does not win everywhere.

## 10. Architecture

```
React (Vite) + Tailwind + Recharts
        |  REST/JSON
Node/Express -- validation, caching, reads precomputed results
   |                     | live calls: /recommend, /simulate
   v                     v
PostgreSQL          FastAPI (Python) -- trained XGBoost + SHAP explainer
 ^  (Docker locally,          ^
 |   Supabase to deploy)      |
 +-- offline pipeline (/ml): data -> features -> train -> backtest -> simulate -> write results
```

- Vite + React (not Next.js): Express is already the backend.
- Precompute forecasts, SHAP, backtest metrics and default simulations into Postgres; FastAPI serves only live what-ifs. Service-level and lead-time selectors snap to the trained grids (alpha in {0.80, 0.90, 0.95, 0.99}, L in {3, 7}); no interpolation between quantile models.
- Dashboard pages: Overview, Demand Forecast, Product Analysis (SHAP), Inventory, Risk/Alerts, Model Performance, Policy Simulator (clearly labelled as historical simulation with stated assumptions).

## 11. Scope

- **MVP:** everything above except the stretch items.
- **Stretch:** conformal calibration, backorder variant, random-lead-time sensitivity, deployment.
- **Out of scope:** price/promotion what-if sliders, weather APIs, drift monitoring, LSTM/Transformers, LLM chatbot, hierarchical reconciliation, ARIMA.

Environment: **Python 3.12** virtualenv with pinned requirements; Node 24; PostgreSQL via Docker.

## 12. Decision log

- 2026-09-20 Approved Phase 1 with amendments: (1) headline = matched-service-level inventory trade-off curve, cost as a sensitivity sweep since default costs imply about 99% critical ratio; (2) service-level grid {0.80, 0.90, 0.95, 0.99}, integer order-up-to levels (ceil, applied to all policies); (3) future-window covariate aggregates and stated known-prices assumption; (4) scaled metrics (WAPE, MASE, scaled pinball) and a target-normalization experiment; (5) point-policy sigma from out-of-sample calibration residuals, naive rule defined now; (6) Python 3.12.
- Choices made while writing this file (open to change): `ceil` rounding; primary matched-service panel = fill rate; RMSE (not std) for sigma; test window 26 weeks, calibration 12 weeks, retrain every 8 weeks; random lead time Uniform{2,3,4}; quantile set includes 0.10/0.50/0.90 for display alongside the four service-level quantiles.
- 2026-09-20 Second round of amendments: (1) model staleness is 12-20 weeks (12-week calibration gap plus up to 8 weeks of use), not 12; (2) velocity terciles and segments are computed once from data up to the tuning cutoff only, and used for sampling, pooling and reporting; (3) bootstrap is a cluster bootstrap over items, not series; (4) per-series sigma and conformal offsets are replaced by scale-normalised values pooled by velocity segment, for both policies; (5) test-window regime documented (holiday peak, closed-store days, regime-lagged calibration), with fixed sub-period reporting and a closed-day calendar feature.

## 13. Left to settle in later phases

- Phase 4: exact feature list, trailing-window lengths, price-median window.
- Phase 6: normalization scale floor; XGBoost hyperparameters (tuned on the rolling tuning folds).
- Phase 9: how the API exposes the "stockout risk" label (HIGH/MEDIUM/LOW) thresholds.
- Phase 3 (EDA): confirm closed-store days and encode them as a calendar rule from training years; compute velocity terciles from data up to the tuning cutoff only.
- Phase 9: residual-vs-scale diagnostic for pooled sigma (constant-CV assumption).
- Phase 10: the `c` grid for the naive rule, backorder variant.
