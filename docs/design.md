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
| `sales_train_evaluation.csv` | wide format, one row per item-store series; id columns (`id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`; `id` is a composite key such as `FOODS_1_001_CA_1_evaluation`) and daily unit sales `d_1..d_1941` |
| `calendar.csv` | date, weekday, month, year, `wm_yr_wk` (price join key), `event_name/type_1/2`, `snap_CA/TX/WI` |
| `sell_prices.csv` | weekly `sell_price` per store and item, keyed by `wm_yr_wk` |

**Sales data spans 2011-01-29 to 2016-05-22** (`d_1..d_1941`; 22 May 2016 is a Sunday). `calendar.csv` runs 28 days further, to 2016-06-19 (1,969 days, the M5 evaluation horizon); those days have no sales and are never used. Verified on load in Phase 2.

**Subset:** state CA, stores CA_1..CA_3, 100 items sampled with a fixed seed, stratified by department and by sales-velocity tercile so slow and intermittent items are represented. That is 300 series (509,871 daily rows after dropping 72,429 pre-launch rows; 582,300 before). The eligible pool is items on sale in **all three** CA stores for at least 2 years (730 days) before the tuning cutoff (the first `sell_price` week starts on or before `2015-08-30 - 730 days`). Measured in Phase 2/3: 2,215 of 3,049 items (2,221 under a 31 Aug cutoff; see the decision log). Everything else is dropped and the count is documented. Scale up only if runtime allows.

**Velocity segments (high / mid / low):** terciles of each series' mean daily sales since its first on-sale week, computed **once, from data up to the tuning cutoff (`test_start - 12 weeks`, section 5) only**. Items are sampled stratified on their mean velocity tercile across the three stores; each series carries its own segment label. The same labels drive the stratified sampling, the pooling of sigma and conformal offsets (section 5), and the result breakdowns (section 9). They are never recomputed with later data, because a segment defined using test-period sales would leak the outcome into how results are grouped and calibrated.

**Cleaning rules known in advance:** rows before an item's first `sell_price` are "not yet on sale", not zero demand, and are dropped. The rule is applied as written to every series, including `FOODS_2_101_CA_2`, whose first sale comes 1,578 days after its first price (kept; its long leading zero run is not special-cased and is noted as a data oddity). Days when the store is closed (Christmas is the known case) are near-zero, not exactly zero: in Phase 2, CA_1..3 total sales across all 3,049 items were 5-8 units on each Dec 25 of 2011-2015, against thousands on a normal day. A closed day is therefore defined by a **near-zero threshold on total store sales** (threshold and the list of closed dates confirmed in Phase 3 EDA), not `== 0`. Such days are flagged, and the calendar rule derived from them is applied to 2015 (section 5).

**Limitations (must appear in the README):**
1. No inventory levels, lead times, or costs in the data. All are simulated assumptions.
2. Sales are not demand: history is censored when stock ran out. We forecast sales as a proxy for demand, and the data is **not filtered** for it. Measured in Phase 3 (training data only): 80 of 100 high-velocity series have a zero run of 14+ open days, about 11% of high-velocity days (15% mid, 34% low) sit inside such runs, and these runs are strongly simultaneous across an item's three stores (see section 12), so part of it is item-level (delisting, supply, seasonal items) rather than store stock-outs. The simulator replays zero demand inside such runs, which flatters every policy on inventory.
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

- **History (all computed from data <= t):** recent daily sales lags, rolling mean/std over 7/14/28/91 days, fraction of zero days, days since last sale, **longest zero run in the last 91 days (capped at 56)**, sum of the last P days, sum of the same window 364 days earlier, current price, item/store/dept/cat ids.
- **Future-window covariate aggregates (window t+1..t+P):** number of SNAP days (state-specific), number of event days (by type where useful), number of weekend days, number of store-closed days (a calendar rule, see section 5), **planned price relative to trailing median** (mean and min window price divided by the trailing median price, the median computed from prices <= t).
- The exact list, windows and a leakage note for every feature are in `docs/features.md` (Phase 4); `ml/features.py` builds it and `ml/test_features.py` checks the leakage rules by perturbing the data after an origin.
- **Known-prices assumption (stated, not hidden):** future prices, events and SNAP days are treated as known at order time. Events and SNAP schedules genuinely are. Prices are an assumption: M5 gives us realised prices, and we treat them as if they were the plan. That flatters ML policies relative to the naive rule, so we run an ablation without future-price features and report the difference.
- Future sales are never a feature. Predictive effects of price are associations, not causal estimates (why "what-if" price sliders are out of scope).

## 5. Splits and leakage rules

Timeline: **test = last 26 weeks.** Concretely, the **first review date is Sunday 2015-11-22** (`test_start`), and **test demand days run Monday 2015-11-23 to Sunday 2016-05-22** (182 days, 26 whole weeks). **Reviews are on Sundays for every series** (the fixed review weekday, section 9); an order placed after close on a Sunday covers demand from the Monday. Model selection and tuning use rolling-origin folds on data up to `test_start - 12 weeks` = **Sunday 2015-08-30** (the tuning cutoff) only.

For a model version with cutoff date `c` (retrained every K = 8 weeks on an expanding window):
- **Fit set:** origins `s` with `s + P <= c - 84 days`. The target window must have fully ended before the cutoff (embargo), otherwise labels leak across the train/test boundary.
- **Calibration window:** the 12 weeks before `c`. Out-of-sample residuals from here give the point policy's sigma and the optional conformal offsets. Same window, same pooling (below) for both policies.
- **Use:** review dates in `[c, c + K weeks)`.
- **Model staleness is 12 to 20 weeks, not 12.** The fit set ends at `c - 84 days` and a version is used for review dates up to `c + 8 weeks`, so the newest data the model has seen is 12 weeks old at the start of its use period and 20 weeks old at the end. Simplification (`ponytail`): the model is not refit on the calibration window. Upgrade path: refit after calibrating (staleness drops to 0-8 weeks), keeping the calibration offsets, at the price of slightly optimistic residuals.
- **Calibration is pooled by velocity segment, never estimated per series alone, for both policies.** A 12-week window gives about 12 weekly origins per series (about 84 daily origins, but overlapping and strongly dependent). That is too few for a reliable per-series sigma, and it cannot support a 0.99 conformal offset at all (a 0.99 quantile needs on the order of 100 scores; the maximum of 12 is not that). Rule: residuals and conformal scores are divided by the series' scale (P x trailing mean daily sales, floored; the same scale as the normalization in section 3), pooled within the velocity segment, and the pooled statistic is multiplied back by the series' own scale: `sigma_s = sigma_segment * scale_s`, `offset_s,alpha = offset_segment,alpha * scale_s`. This assumes a roughly constant coefficient of variation within a segment; Phase 9 checks it with a residual-vs-scale diagnostic (fallback: scale by the square root of the scale, which is Poisson-like). Shrinking each series toward its segment value is the alternative if pooling is too coarse. Same window, same segments, same rule for the point policy's sigma and the quantile policy's offsets, so neither gets better-calibrated inputs than the other.

General rules: never shuffle; nothing (scalers, price medians, scale factors, item sampling thresholds) is fitted on data after the origin; all history features are computed as-of t; test data is never used for tuning; the test-window forecasts are computed once and reused across policies (forecasts depend only on historical sales, not on simulated stock).

**Origins:** an origin is evaluated only if its whole target window lies inside the data, and the simulator needs the full protection interval too. The longest horizon is P = 14, so the **last usable review date is Sunday 2016-05-08, 14 days before 2016-05-22**. That gives **25 Sunday review dates** (2015-11-22 to 2016-05-08). The same review dates are used for every P (including P = 7 and 10, which could go a little later) so results are comparable across horizons and policies; the last two demand weeks are only ever the tail of a protection interval.

**Test window regime (read before interpreting any result).** The test demand days (23 Nov 2015 to 22 May 2016) include the Thanksgiving-Christmas-New Year peak, **closed-store days** such as Christmas Day (near-zero sales; threshold confirmed in Phase 3), the Super Bowl and Easter.
- Results mix a holiday-peak regime with a normal one. We therefore report two sub-periods on demand days, with boundaries fixed now: **holiday peak = 23 Nov 2015 to 3 Jan 2016 (6 whole weeks)**, and **rest = 4 Jan to 22 May 2016 (20 whole weeks)**. Both are whole Monday-Sunday weeks, so each review cycle belongs to exactly one sub-period (assigned by the cycle's first demand day). We do not claim results generalise to other seasons.
- Closed days are known from the calendar, so the future-window closed-day count is a legitimate feature. A closed day is one where total store sales fall below a near-zero threshold (section 2), not `== 0`. It is identified as a calendar rule (e.g. Dec 25) from **training years only** (2011-2014, all before the tuning cutoff), never from sales inside the window being forecast, and then applied to 2015. In the simulator, closed days replay zero demand while holding cost still accrues.
- Calibration windows lag the regime. Version 1's calibration window (autumn, calm) is applied through the holiday peak, so sigma and conformal offsets will likely be too small there; the next version's window contains the peak and will be too large afterwards. **Post-hoc observation (see section 12):** the rest period also sits at a higher volume level than the summer calibration windows, which pushes the same way, so **under-service in the rest period is the expected effect for both policies**. Expect achieved-vs-target service gaps in both policies; conformal prediction carries no exchangeability guarantee for time series.
- **Model versions:** version 0 (cutoff 2015-10-25) serves the four warm-up review dates 25 Oct to 15 Nov; version 1 (cutoff 2015-11-22) serves 22 Nov to 10 Jan (its fit set ends 2015-08-30 = the tuning cutoff); version 2 (cutoff 2016-01-17) and version 3 (cutoff 2016-03-13) follow, and version 4 (cutoff 2016-05-08) serves only the last review date. Version 0's fit set ends 2015-08-02. The first versions were fit on data ending late July to late August 2015, so they have seen only four earlier holiday seasons (2011-2014).

## 6. Metrics (scaled, so series of different volume are comparable)

- **Point:** WAPE = sum|y - yhat| / sum y (pooled). **MASE** per series = test MAE / in-sample MAE of the naive-P forecast (`y_t - y_{t-P}`) on the training period, averaged over series. Series with zero scale are excluded and counted. Raw MAE/RMSE are reported for reference. MAPE is shown only on non-zero targets, to demonstrate that it breaks at zero, and is never used for selection.
- **Probabilistic:** **scaled pinball loss** = pinball loss divided by the same per-series scale as MASE, averaged over quantiles and series. Also **coverage**: empirical share of actuals inside [P10, P90] (nominal 0.80) and one-sided coverage of each quantile (share of `y <= q_alpha` vs alpha). One-sided coverage is what the reorder policy needs. Because demand is a small integer, nominal-or-higher coverage is expected for slow movers, and is explained rather than hidden.
- Evaluation origins are weekly (Sundays). **MASE scale** for a tuning fold is each series' in-sample MAE of the naive-P forecast over all daily origins whose target ended on or before the fold's first origin (nothing from the fold or later). **Point baselines have no quantiles**, so for them the one-sided figure reported is the share of actuals at or below the point forecast (the quantile level the forecast implicitly sits at), labelled as such; interval coverage is reported from Phase 7 on.

## 7. Models

Baselines for the P-day sum: **naive** (previous P days), **MA-28** (28-day open-day mean x P, i.e. the 28-day mean scaled to the P-day window), **seasonal naive** (same window 364 days earlier, weekday-aligned; where that window does not exist because the series is under a year old it falls back to naive-P, and the share of fallback rows is reported). "Same weekday last week" collapses into naive for a 7-day sum, hence the swap. Then Linear Regression, Random Forest, XGBoost (squared error = mean forecast), XGBoost quantile (`reg:quantileerror`, LightGBM as fallback). ARIMA is dropped. Conformal calibration (split conformal on the calibration window, scores pooled by velocity segment as in section 5) is a cheap stretch, reported as coverage before vs after.

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
| Policy | Periodic review R = 7, order-up-to S, review weekday fixed to Sunday for all series | Real retailers stagger reviews by supplier |
| Lead time | Fixed L = 3 base (grid {3, 7}) | No supplier data; random lead time is a sensitivity |
| Timing | Order placed after close of day t, arrives before opening day t+L+1, so the protection interval is exactly days t+1..t+R+L, matching the target | |
| Start | Simulation starts at the review on Sunday 2015-10-25 with stock at S, using the **version-0 model**. The four warm-up cycles (25 Oct to 15 Nov) are excluded from metrics, so all 25 review dates from 22 Nov to 8 May count | Orders in flight at 22 Nov come from the warm-up; warm-up forecasts use the same rules as the test |
| Prices/costs | Unit price = item-store median price in the test window; unit cost = 70% of price; holding = 25%/year of unit cost per unit-day; stockout = lost margin (30% of price). Fixed before any results. | Illustrative; not from data |

**Headline result: trade-off curves at matched service levels.** For each policy (naive over `c`, point and quantile over alpha), plot **average on-hand inventory (units) against achieved service**. Two panels: **fill rate** (units served / units demanded, pooled; the business-facing metric and the primary panel) and **cycle service level** (share of review cycles without a stockout; what the quantile target alpha nominally controls, so also a calibration check of the decisions). Comparison = inventory needed to reach the same achieved service, linearly interpolated between points, no extrapolation, with the coarseness of a four-point grid stated. 95% confidence intervals from a **cluster bootstrap over items**: an item's three store-series are resampled together because they share the item's demand drivers and are not independent, so the effective sample size is 100 items, not 300. The bootstrap does not capture period-to-period variability (there is one 26-week test period), so the intervals understate uncertainty about other seasons; the README says so. Results are also broken down by velocity segment (high / mid / low, defined in section 2) and by the two test sub-periods (section 5).

**Cost metric is a sensitivity sweep, not the headline.** With the default costs, the cost-implied critical ratio is: holding per unit over one 7-day cycle = 0.7 x 0.25/365 x 7 x price = about 0.0034 x price; stockout = 0.3 x price; ratio = about 89, critical ratio about 0.989 (about 99%). So at default costs a cost-optimiser would simply choose the top of the grid, and a cost "win" would say more about the cost assumption than the policy. Instead we sweep the stockout-to-holding cost ratio rho over {4, 9, 19, 99}, whose critical ratios are {0.80, 0.90, 0.95, 0.99}, and report each policy's minimum total cost over its own settings at each rho. (Newsvendor critical ratio is a textbook approximation for periodic-review lost sales; the simulation is the arbiter.)

**Other sensitivities:** lead time L in {3, 7}; random lead time L ~ Uniform{2, 3, 4} (same mean 3 as the nominal, isolates variance; policies still plan with nominal L; orders may cross); no-future-price ablation; optional backorder variant; and a hindsight diagnostic, **"exclude zero-demand cycles"**: drop, from every policy's metrics alike, the review cycles in which the series' realised demand over the protection interval was zero. It is defined with hindsight, so it is a diagnostic of how much stock-out-like history distorts the comparison and never the headline.

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
- 2026-09-20 Phase 2 data verification (M5 files loaded; all counts, columns and dates match section 2) and resulting decisions, each logged separately:
  1. **Review calendar.** First review date is Sunday 2015-11-22; test demand days are 2015-11-23 to 2016-05-22; the review weekday is Sunday for all series; holiday peak = 23 Nov to 3 Jan (6 whole weeks), rest = 4 Jan to 22 May (20 whole weeks). The last usable review date is 2016-05-08 (14 days before the last sales day, because targets and simulation need the full P = 14 protection interval), giving 25 review dates common to all horizons. This replaces the earlier "about 22 Nov" wording, since exactly 26 weeks ending 22 May starts on Monday 23 Nov. Consequence: `test_start` is the first review date, so the tuning cutoff is Sunday 2015-08-30.
  2. **Closed days** are defined by a near-zero threshold on total store sales, not `== 0` (Dec 25 showed 5-8 units across CA_1..3 in every year 2011-2015). Threshold and closed-date list confirmed in Phase 3 from training years only (2011-2014), then applied to 2015 as a calendar rule.
  3. **Two-year history rule** uses the tuning-cutoff reading (2 years of history before the tuning cutoff, all three CA stores on sale), not "before the test window". With the 2015-08-30 cutoff from decision 1 this leaves **2,215** eligible items (2,221 was measured with a 31 Aug cutoff derived from a 23 Nov test start; 2,306 under the looser test-window reading). Enough to sample 100 items.
  4. **Documented data differences:** the sales file has a composite `id` column not listed originally; `calendar.csv` runs to 2016-06-19 (28 days past the last sales day) and those days are unused.
- 2026-09-20 Phase 3 (`ml/prepare.py`, `ml/eda.py`, figures in `docs/figures/`). Cleaning and subset, all derived from data <= 2015-08-30:
  - **Closed-day rule** = Dec 25: in 2011-2014 every CA store had a total below 1% of its median daily total on that date (max 7 units, against a minimum of 1,150 on any other training day), and no other date qualified. Applied to 2015 as a calendar rule. (2015-12-25 does show 0/2/4 units, consistent with the rule, but it was not an input.)
  - **Velocity segments:** series tercile cuts at 0.514 and 1.404 units/day (closed days excluded). Sampled items by stratum 33 low / 34 mid / 33 high, allocated across department x tercile cells (minimum 1 per cell, rest proportional); the resulting series segments are 99 low / 101 mid / 100 high. Departments: FOODS_3 25, HOUSEHOLD_2 19, HOUSEHOLD_1 18, HOBBIES_1 16, FOODS_1 10, FOODS_2 6, HOBBIES_2 6 items. Seed 42.
  - **Price gaps:** none. Every on-sale day of the 300 series has a `sell_price`, and no sale occurs without a price.
- 2026-09-20 Phase 3 findings that qualify earlier assumptions (open decisions in section 13):
  1. **Long zero runs look like stock-outs or delistings, not demand.** In training, 80 of 100 high-velocity series (3.45 units/day on average) have at least one zero run of 14+ days after their first sale (median run 28 days); about 11% of high-velocity days, 15% of mid and 34% of low sit inside such runs. This is limitation 2 (censored sales) measured: a series selling 3-4 units a day does not go 14 days without a sale by chance. As a result even 9% of 14-day windows are all-zero for the high segment (11% at 7 days), against 30% of days.
  2. **The holiday peak is small in volume.** For series on sale since mid-2011, mean daily units in 23 Nov to 3 Jan ran x1.00, 1.08, 1.01, 1.10 and 1.07 of the Sep-Oct level in 2011-2015. The main holiday effect on the data is the Dec 25 closure (finding 3 covers where the bigger level shift is).
  3. **Level in the test window is above the calibration-window level.** Series on sale since mid-2011 averaged 1.24 units/day in the 2015 calibration-like window (8 Jun to 30 Aug) and 1.43 (x1.16) in the holiday peak and 1.49 (x1.20) in the rest of the test window. In 2011-2014 the peak ran x0.91 to x1.13 and the rest x0.80 to x1.04 of the same summer window. So the regime lag section 5 warns about is real, but it is a spring-2016 level rise, not the holiday peak. Sigma and conformal offsets are scaled by the trailing mean, which absorbs part of it.
  4. **Weak, associational effects** (not causal): Saturday/Sunday index 1.23/1.19 vs about 0.88 midweek; FOODS units on SNAP days 2.23 vs 2.06 other days (SNAP days are 33% of days, and they cluster early in the month); on days priced 15%+ below the series median, sales index 1.38 vs 1.01 at the usual price, and 0.81 when priced 5%+ above. Prices change weekly; 82% of days are within 5% of the series median.
  5. **Intermittency:** daily zero share 80% (low), 55% (mid), 29% (high). Syntetos-Boylan classes: low 89 intermittent / 10 lumpy; mid 77 / 24; high 27 smooth / 10 erratic / 28 intermittent / 35 lumpy. Zero-window share falls with P: low 80% / 36% / 22% at P = 1 / 7 / 14. This supports the P-day-sum target.
  6. **Dead or near-dead series:** 28 of 300 series had zero sales in the 28 days before the tuning cutoff (12 high, 10 mid, 6 low; consistent with finding 1), and 1 series had zero sales over the whole test window. `FOODS_2_101_CA_2` has its first sale 1,578 days after its first price, so its first-price date is not a true launch.
- 2026-09-20 Decisions after Phase 3, each logged:
  1. **Stock-out runs:** the data stays unfiltered and limitation 2 is quantified (section 2). A "longest zero run in the last 91 days, capped at 56" feature is computed as of t (the cap of 56 is a Phase 4 choice, adjustable on the tuning folds only). Phase 10 reports a hindsight "exclude zero-demand cycles" diagnostic applied equally to all policies (section 9), not as the headline.
  2. **Cross-store check of zero runs** (training data, runs of 14+ open days; 2,125 runs; 19.9% of on-sale series-days are inside one). Simultaneity is far above independence: of run-days, 30.1% have only this store out (64.1% if independent), 29.5% two stores (31.9%), 40.4% all three (4.0%). 28.1% of runs never overlap another store's run, 61.9% overlap on at least half their days, 34.0% are fully overlapped. By segment, all three stores out on 34.7% (low), 46.9% (mid), 47.8% (high) of run-days. Reading: a large share is item-level (delisting, supply, or genuinely seasonal items, which are also zero everywhere at once) and a substantial minority is store-specific. This check cannot separate seasonal zeros from supply failures; that was not tested, so "stock-out" in Phase 3 finding 1 should be read as "stock-out, delisting or seasonal absence". No data was changed.
  3. **`FOODS_2_101_CA_2` is kept** under the pre-registered first-price rule and noted in section 2.
  4. **Warm-up:** the simulation starts at the 2015-10-25 review with a version-0 model, so all 25 review dates from 22 Nov to 8 May count in the metrics (sections 5 and 9). This replaces the earlier "4-week warm-up excluded" and resolves the Phase 10 question.
  5. **Disclosure about the test window:** the Phase 3 EDA looked at test-window volume (`weekly_level.png` and the level comparison by year include it). The holiday-peak / rest split (23 Nov to 3 Jan / 4 Jan to 22 May) was fixed before that and is **unchanged**. The spring 2016 level rise (about 1.2x the summer calibration-window level, against 0.80-1.04x in earlier years) is a **post-hoc observation**, not a pre-registered hypothesis; its expected effect is under-service in the rest period for both policies (section 5). From here on every modelling choice, including target normalization, windows and caps, is decided on the tuning folds only.
- 2026-09-20 Phase 5 pre-registration, logged **before any baseline score was computed** (folds in `ml/folds.py`, committed with this entry):
  1. **Tuning folds.** Four non-overlapping folds of 12 consecutive Sunday review origins each, all on data up to the tuning cutoff (the last fold's P = 14 targets end exactly on 2015-08-30): **F1** 2014-08-31 to 2014-11-16 (autumn, Thanksgiving in its targets); **F2** 2014-11-23 to 2015-02-08 (the holiday fold: the same calendar season as the test window's holiday peak, a year earlier, spanning Thanksgiving, Christmas with its closure, New Year and the Super Bowl); **F3** 2015-03-08 to 2015-05-24 (spring); **F4** 2015-05-31 to 2015-08-16 (summer, the season of version 1's calibration window). The same origins serve P = 7, 10 and 14. Chosen by a fixed rule (12 Sundays, last fold anchored to the cutoff, one fold aligned to the test window's calendar season) and not from any score. Contiguous except a three-week gap between F2 and F3.
  2. **What is run:** naive, MA-28 and seasonal naive on F1-F4 for P in {7, 10, 14}: WAPE (pooled, and by velocity segment), MASE (per-series scale as in section 6, zero-scale series excluded and counted), MAE, RMSE, and the share of actuals at or below the point forecast. Results are written to `docs/results/` with the commit that produced them.
  3. **No test-window metric is computed.** `folds.assert_tuning_only` enforces it in code (a fold whose target windows pass the tuning cutoff raises), `test_folds.py` tests the guard, and section 14 logs every touch.
  4. **Testing is part of the definition of done** for every phase from Phase 5 on (section 15, `docs/testing.md`), CI runs the fast suite.

## 13. Left to settle in later phases

- Phase 4 is done: feature list, windows and the 182-day price-median window are in `docs/features.md`. Modelling choices in it (windows, caps, normalization later) are decided or changed on the tuning folds only.
- Phase 6: normalization scale floor; XGBoost hyperparameters (tuned on the rolling tuning folds).
- Phase 9: how the API exposes the "stockout risk" label (HIGH/MEDIUM/LOW) thresholds.
- Phase 3 is done (closed-day rule, terciles, subset; see the decision log). Open from it:
  - **Dead series at the cutoff (Phase 6):** 28 series had zero sales in the 28 days before the cutoff; the normalization scale floor must handle them.
- Phase 10: implement the warm-up (version-0 model, simulation from 2015-10-25) and the hindsight "exclude zero-demand cycles" diagnostic as decided in section 12.

## 14. Test-window touch log

Every time anything (a metric, a plot, a table, a model comparison, a threshold, a hyperparameter) is computed from data whose demand day falls in the test window (2015-11-23 to 2016-05-22), it is recorded here with date, what was computed, and whether it informed any decision. Tuning-stage code calls `folds.assert_tuning_only`. Model versions that retrain during the test window (section 5) use only data before their own cutoffs and are not touches.

| Date | Phase | What was computed on test-window data | Informed a decision? |
|---|---|---|---|
| (no entries) | | | |

The Phase 3 EDA looked at test-window *volume* before this log existed; that is disclosed in section 12 (decision 5 after Phase 3) and is not repeated here.

## 15. Testing and definition of done

From Phase 5 on, a phase is complete only when its tests are written and pass: the fast suite (synthetic fixtures, no data, run by CI on every push) and, where marked, the real-data checks (`pytest -m realdata`). Tests assert definitions, invariants, hand-computed values and synthetic ground truth, never unmeasured accuracy. What each component must be tested for, and what is still pending, is in `docs/testing.md`; a skipped placeholder there blocks its phase.
