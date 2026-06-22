# Probabilistic Demand Forecasting & Inventory Policy Simulator

Forecasts the **distribution** of demand over a protection interval (7, 10 or 14 days) for retail item-store series, and is designed to compare
inventory ordering policies that use those forecasts in a historical replay. Built on the M5 (Walmart) data, 300 California series.

**Status: work in progress.** Data preparation, features, baselines, point and quantile models, their evaluation on the tuning folds, the explanation
layer and the planner-facing quantities (reorder point, safety stock, stockout-risk label) are done (phases 2 to 9). The base-case inventory replay, its sensitivities and the forecast-accuracy report have been run on the test window (phase 10, frozen as **v1**). A static dashboard and an Express API serving those precomputed results are built and tested locally (see *The demo* below); **they are not deployed yet**.
**Inventory results are historical replays under stated assumptions (no real stock, lead-time or cost data exist), not claims about a real retailer.** Every number in this file comes
from a logged run; the decision log and the pre-registered rules are in [`docs/design.md`](docs/design.md), the run outputs in [`docs/results/`](docs/results/).

## Evidence of forecast skill: the test-window forecast metrics

The only skill evidence in this project is the forecast accuracy on the **held-out test window** (2015-11-23 to 2016-05-22), computed once from the frozen models ([`docs/results/phase10_test_forecast_metrics.md`](docs/results/phase10_test_forecast_metrics.md)): 25 review dates x 300 series, mean over model versions v1 to v4. Tuning-fold numbers below are model-selection results and are optimistic; inventory comparisons are descriptive (see the null result next to them).

| Metric (test window) | Horizon 10 days | Horizon 14 days |
|---|---|---|
| **Mean scaled pinball, quantile model vs the post-hoc benchmark B3a** (same point forecast + empirical residual quantiles; lower is better) | **0.2296 vs 0.2706**; difference -0.0410, 95% interval [-0.0504, -0.0337]; lower in 4 of 4 versions | **0.2444 vs 0.2985**; difference -0.0542, 95% interval [-0.0642, -0.0460]; lower in 4 of 4 versions |
| Same, against B3c (MA-28 + empirical residual quantiles, post-hoc) | -0.0591 [-0.0714, -0.0495] | -0.0734 [-0.0854, -0.0635] |
| **Coverage of the ordered quantity** (share of demand at or below ceil(q)), nominal 0.80 / 0.90 / 0.95 / 0.99 | 0.843 / 0.926 / 0.960 / 0.994, consistent with nominal at all four levels | 0.823 / 0.904 / 0.953 / 0.993, consistent at all four levels (a few high-velocity cells under-cover at 0.80) |
| **Mean-forecast WAPE, normalised XGBoost mean vs MA-28** (like for like: both estimate the mean) | 0.3436 vs 0.3504: roughly on par (2% lower) | 0.3146 vs 0.3199: roughly on par (1.7% lower) |
| Median-forecast WAPE, the quantile model's median vs MA-28 (see the caveat below) | 0.3189 vs 0.3504 | 0.2953 vs 0.3199 |

- **The median caveat.** The quantile model's median is 9% below MA-28's WAPE at horizon 10 (8% at horizon 14), but **the median is the forecast that minimises absolute error, and WAPE is an absolute-error metric**, while MA-28 and the XGBoost mean forecast estimate the mean. Part of that gap is a difference of objective, not evidence that demand is found better; the like-for-like point comparison is the mean forecast against MA-28, and it is on par.
- **Horizons.** Test-window metrics exist for horizons 10 (the base case) and 14 (the lead-time-7 sensitivity). **There are no test-window metrics for horizon 7**; horizon 7 has tuning-fold results only.
- **Uncertainty.** Intervals are a bootstrap over the 100 items: they reflect which items were sampled, not variation between periods (one 26-week window, one retailer, 300 series from three stores). B3a and B3c are post-hoc benchmarks defined after the tuning results were known; against the pre-specified normal-sigma policy (B2, mean scaled pinball 0.3783 at horizon 10) the gap is larger, mostly because of the normal-sigma assumption.
- **Reproducibility.** The data pipeline and the tests reproduce from a clean clone (features exactly equal). **The fitting stages from Phase 6 onward were not reproduced from cold** (hours of fits; the caches are git-ignored). The results rest on **frozen models whose SHA-256 hashes were recorded when they were saved and re-verified: 34 of 34 files match** ([list](docs/results/model_hashes.md)); the serving export is hash-verified against `models.lock.json` at start-up. One cheap version (development fold F4, horizon 10) was **retrained from scratch and reproduced the saved forecasts exactly** (maximum difference 0 units, 6,000 trees in both; same machine, thread count and library versions; [result](docs/results/phase11_retrain_check.md)); the test-window versions were not retrained.

**Documentation:** [architecture](docs/architecture.md), [methodology](docs/methodology.md), [limitations](docs/limitations.md), [resume bullets and claims to avoid](docs/resume.md), [interview Q&A](docs/interview_qa.md), the decision log [`docs/design.md`](docs/design.md).

## What was evaluated, and how

- **Target:** cumulative demand over the next P days from a Sunday review date, for P in {7, 10, 14}, as a full quantile forecast (0.10, 0.50, 0.80, 0.90, 0.95, 0.99).
- **Data:** stores CA_1 to CA_3, 100 items sampled with a fixed seed and stratified by department and sales velocity: 300 series, 2011 to 2016.
- **Models:** naive, MA-28 and seasonal-naive baselines; Linear Regression, Random Forest and XGBoost for point forecasts; XGBoost quantile regression; optional conformal calibration.
- **Honest evaluation:** all model choices were made on four rolling-origin *tuning folds* that end before 2015-08-30, with rules committed to `docs/design.md` **before** scoring. The **test window (2015-11-23 to 2016-05-22) has been used for one base-case inventory replay, its pre-registered sensitivities, the test-window forecast metrics and a few labelled diagnostics**; every use had its touch-log entry written first, the code was frozen before the replay, and every use is listed in the touch log.
- **Tests:** a fast synthetic suite (run by CI) plus real-data checks; a phase is done only when its tests pass and are recorded ([`docs/testing.md`](docs/testing.md)).

## Model-selection results (tuning folds; not the skill evidence)

**Tuning-fold headline (model selection; the test-window equivalent is in the evidence section above).** Against the post-hoc benchmark that keeps the same point forecast but takes its quantiles from empirical residuals, the quantile model has **about 13% lower scaled pinball on the tuning folds** (0.2639 vs 0.3025, better in 12 of 12 fold-by-horizon cells, difference -0.039, 95% interval [-0.049, -0.031] over items). **The normal sigma was most of the gap to the textbook policy** (a normal quantile around the mean forecast with a pooled RMSE sigma, 0.4128): replacing it removes about three quarters of that gap. The gap to the textbook policy is never quoted without this.

**Point forecasts** ([phase 6](docs/results/phase6_point_tuning.md), [bootstrap](docs/results/phase6_point_bootstrap.md)). Against the MA-28 baseline, pooled WAPE ratio at P = 7 / 10 / 14: Random Forest 0.962 / 0.957 / 0.958 (better in all 12 fold-by-horizon cells; difference -0.0157, 95% interval [-0.0228, -0.0085] over items), XGBoost 0.985 / 0.967 / 0.978 (11 of 12 cells; -0.0091 [-0.0164, -0.0010]), Linear Regression about 1.00 (no difference). The gain comes from mid- and high-velocity series: for low-velocity series no learned point model beat MA-28. **Read these as small gains, and note which model they describe:** the numbers above are for the raw-target models of Phase 6. The pipeline behind the quantile forecasts and the B2 / B3a policies uses the *normalised-target* XGBoost mean forecast, which is **roughly on par with MA-28**: 4% to 5% worse on WAPE in the tuning folds (ratio 1.038 / 1.043 / 1.054, Phase 6) and 2% better on the test window (0.344 against 0.350 at horizon 10). Item-store demand is noisy (a Poisson-noise floor computed around the forecast, [table](docs/results/phase9_noise_floor_tuning.md), is about half the observed error, and real demand is noisier than Poisson), so point accuracy is not where this project adds value. **The value is in the calibrated quantile forecast and the inventory replay.**

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

**Forecast accuracy on the test window:** see the evidence section at the top (horizon 10 and 14, with intervals, coverage and WAPE). One result went against my expectation: the pipeline's normalised mean forecast was 2% better than MA-28 on the test window, not worse as in the tuning folds, so it is described as on par.

**Inventory replay on the test window: descriptive comparisons at matched fill rate** ([phase 10 primary run](docs/results/phase10_simulator_primary.md), horizon 10 / lead time 3 days, weekly Sunday review, lost sales, 25 review dates, one run made once). The **fair benchmark is B3a**, so every comparison leads with it. **Read every reduction below together with the shuffled-demand null result:** with realised demand shuffled so that it is unrelated to the forecasts, the quantile policy still shows a reduction against B3a ([test window](docs/results/phase11_null_test_test.md): 7.5% with each series given another series' demand, 27.0% with demand shuffled in weekly blocks, against 16.0% with the real demand; [tuning folds](docs/results/phase11_null_test_tuning.md): 9.7%, 13.0% and 3.9%). The comparison mixes forecast skill with the different shapes of the two rules' order-up-to levels, so it is **descriptive and is not skill evidence**.

- **Against B3a (comparator-anchored: the quantile policy's inventory interpolated at each B3a setting's own fill rate), at B3a's 0.80, 0.90 and 0.95 settings, the quantile policy needs 3.5% to 9.5% less inventory at the same fill rate.** The primary run gives 4.5%, 7.7% and 5.6%; the range is over the primary run and the four sensitivity variants (lead time 7, random lead time, no future-price inputs, zero-demand cycles dropped), at every matched setting between 0.80 and 0.95.
- **How sure:** the primary run's pooled interval, over all four B3a settings, is [0.7%, 24.9%] (mean 16.0%, dominated by the 0.99 setting at 46.3%; item bootstrap, so it reflects which items were sampled, not variation between periods). The quantile-anchored statistic (B3a's inventory interpolated at the quantile policy's own fill rate; pre-registered) gives 12.7%, [9.5%, 24.4%], matching only two settings. Both exclude zero, the first narrowly. The low- and high-velocity segments and the holiday peak include zero, and the individual moderate settings were not bootstrapped, so **an advantage at moderate service levels is suggestive, not established**.
- **The null result in full:** a shuffled-demand null test ([test window](docs/results/phase11_null_test_test.md), [tuning folds](docs/results/phase11_null_test_tuning.md)) gives a *positive* matched-inventory reduction against B3a even when realised demand is unrelated to the forecasts (5% to 10%), and a larger one when only the timing is destroyed (13% to 27%), because the two rules' order-up-to levels have different shapes. **The reductions above are therefore descriptive comparisons of two rules at equal fill rate under real demand, not evidence of forecast skill;** the evidence of skill is the pinball and coverage results above.
- **0.99 is the costly tail for every policy.** The quantile policy needs 37.7 units per series-day for a fill rate that B3a reaches with 69.7. Going from 0.95 to 0.99 costs the quantile policy 10.9 extra units per fill-rate point and B3a 30.6, so 0.99 is inefficient for the quantile policy (predicted, confirmed) and more so for B3a.
- **Against the other policies** (comparator-anchored): about 29% less inventory than the pre-specified normal-sigma point policy (B2) and about 45% less than the naive c x average-demand rule. B2's normal-sigma assumption failed the constant-CV check for high-velocity series, so its deficit there partly measures that assumption.
- **Predictions committed before the run:** 0.99 inefficient for the quantile policy (confirmed); a smaller edge over B3a than over the point policy (holds pooled and for high velocity, fails for mid velocity, so not confirmed); under-service in the rest period (refuted: the quantile policy served above its target at every level in both sub-periods).
- **Sensitivities** (each run once, expected outcomes written first; [results](docs/results/)): 10 of 13 expected outcomes held and 3 did not, all three about a pooled mean that shifts with which of the four settings could be matched. A follow-up check of random lead time at 0.99 found the draws were applied (3,576 of 5,394 orders had a lead other than 3) and the fill rate there (0.99576 fixed, 0.99576 random mean) moves less than the draw-to-draw spread; cycle service did fall.
- **Cost: unresolved.** The naive rule had the lowest total cost at the cost ratio for a 0.80 target (its minimum is interior after a post-hoc extension of its grid), but the other policies' minima sit at their 0.80 grid edge and cannot be extended, so the comparison of best cases is unresolved.

**Planner quantities** ([phase 9](docs/results/phase9_policy_quantities_tuning.md)). Reorder points and safety stock from the quantile model are close to nominal coverage at 0.80 to 0.95 (0.844 / 0.917 / 0.960), with mean safety stock 4.1 / 7.5 / 11.3 units; the classic normal-sigma version over-protects at every level below 0.99 (coverage 0.944 / 0.969 / 0.979; safety stock 28.2 / 42.7 / 54.7), mostly in high-velocity series. The constant-CV assumption behind the classic sigma **fails the pre-registered check** for high-velocity series (all 12 cells) and half the mid-velocity cells, so the classic figures are not a general statement about textbook policies. The stockout-risk label (HIGH / MEDIUM / LOW plus an overstock flag) orders realised shortfall frequency correctly, but it is a **risk band, not a probability**. Investigated ([label investigation](docs/results/phase9_label_investigation_tuning.md)): no boundary bug, and the median itself is well calibrated. HIGH rows ran short less often than the label suggests where the median is below one unit (rounded up to 1: shortfall 0.17) and inside a long zero run (an outage whose end the model cannot know); where the median exceeds 2 units, HIGH rows ran short 54% to 68% of the time. Any display of the label carries those two qualifiers. No cost or service result exists yet.

Planner sentences do not interpret the series-age input (it sits in a neutral "remaining factors" theme); attributions are associations, not causal effects.

## The demo: precomputed results, no live model

The deployed demo (not yet deployed) **serves precomputed results only**: Postgres holds the scenario results and, locally, the per-series quantile tables; an Express API reads them; a static page renders them. **No model runs behind the public demo.** The order-quantity what-if is `max(0, ceil(q) - inventory position)`, computed in Express from a stored quantile. The FastAPI model service stays in the repository (`api/`, its tests, `docker-compose.yml` with a hash-verified, natively exported model) but is not part of the deployed demo.

- **Scenarios shown are only those that were run** (base case, lead time 7, random lead time, no future-price inputs, zero-demand cycles dropped). Every comparison leads with B3a, with its interval and its statistic label; the 0.99 setting is marked as the costly tail; risk labels are bands, not probabilities; cost is shown as unresolved; there are no test-window explanations (those wait for the final frozen explanation run).
- **Cold starts.** The page renders at once from a recorded snapshot (`web/snapshot.json`), says so, and swaps to live data when the API answers (8 s timeouts, a "waking the service" message).
- **Data publication.** The per-series quantile tables are derived from M5 sales and are loaded only into a local database (the loader refuses a non-local one) until the Kaggle data-use terms are confirmed; the committed snapshot holds aggregates only (a test checks that it contains no series identifiers or dates).
- **Health and uptime (availability only; drift monitoring is out of scope).** `GET /health` is a cheap liveness check returning the version. `GET /health/deep` checks the database (`SELECT 1`) and that the loaded precomputed-data version is the one the code expects; it returns `503` with a short reason (`database` or `data version`) and nothing else. `GET /api/status` reports the time and outcome of the last deep check (the dashboard footer). Requests are rate-limited (120 a minute, 30 on `/health/deep`). **Uptime monitoring status: not set up.** The intended monitor is one UptimeRobot HTTP check on `/health/deep` every 5 minutes with email alerts, created by the repository owner after deployment (never on heavy routes); the badge will be added here when it exists.
- **Run it locally:** `docker compose up -d db`, load with `node scripts/load.js` (in `server/`, `DATABASE_URL=postgres://demo:demo@127.0.0.1:5432/demo`, add `--quantiles` after `uv run python ml/build_demo_data.py quantiles`), `docker compose up -d server`, open `web/index.html` (set `API_BASE` in `web/config.js`). Tests: `uv run pytest` and `cd server && npm test`.

## Known limitation and future work

**Everything here is v1 and frozen.** Future work, to be run only as a separately pre-registered **v2 reported next to v1, never replacing it:** dropping the series-age input (`age_days`; a post-hoc, report-only tuning-fold ablation lowered scaled pinball from 0.2639 to 0.2594, -1.7%, and the model is unchanged), training on more series than 300, a Tweedie or negative-binomial loss, and a wider hyper-parameter search.

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
uv sync --group api
uv run pytest                    # fast suite, no data needed
uv run python ml/prepare.py      # clean the data, derive the closed-day rule and velocity segments, draw the 100-item subset
uv run python ml/features.py     # features and targets
uv run pytest -m realdata        # real-data checks
uv run python ml/build_demo_data.py   # web/snapshot.json from the committed result files
(cd server && npm ci && npm test)     # Express API tests
```
