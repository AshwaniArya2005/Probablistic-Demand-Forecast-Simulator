# Interview Q&A

Short, honest answers with the numbers a listener can check. Forecast figures are the test-window ones ([`results/phase10_test_forecast_metrics.md`](results/phase10_test_forecast_metrics.md)); always give the horizon.

**1. What did you build?**
A pipeline that forecasts the full distribution of demand over the next 7, 10 or 14 days for retail item-store series (300 M5 series), a replay simulator that compares ordering rules using those forecasts, and a small demo that serves the precomputed results. The forecasts are quantiles (P10 to P99) from one XGBoost multi-quantile model per horizon.

**2. Why quantiles rather than a point forecast?**
An order-up-to policy needs a level, not a mean: the quantile at the target service level is that level. A point forecast plus a normal-sigma buffer is the textbook approach and it assumes normal, constant-variance demand; item-store demand is small, integer and skewed.

**3. What is the headline result, and how solid is it?**
On a held-out 26-week window at **horizon 10**, mean scaled pinball loss is 0.2296 against 0.2706 for the empirical-residual benchmark B3a (15% lower; difference -0.0410, 95% bootstrap interval [-0.0504, -0.0337]; lower in 4 of 4 model versions), with coverage of the ordered quantity consistent with nominal at 0.80 to 0.99. At **horizon 14** it is 0.2444 against 0.2985. Solid for those items and that window; the interval reflects item sampling, not other seasons, and there is one window and 300 series.

**4. B3a is a post-hoc benchmark. Isn't that cherry-picking?**
It was defined after the tuning results were known, and I say so everywhere. The point of it is the opposite of flattering: the pre-specified textbook policy loses by a lot (test window 0.3783), but most of that is its normal-sigma assumption. B3a keeps the same point forecast and takes quantiles from empirical residuals, so it isolates what the quantile model adds. I always lead with B3a and label the 36% to 39% gap to the textbook policy as mostly the sigma assumption.

**5. Your mean forecast is no better than a 28-day average. Why use machine learning at all?**
That is what the data show: on the test window the mean forecast's WAPE is 0.3436 against 0.3504 for MA-28 at horizon 10 (on par), and slightly worse in the tuning folds. Item-store demand is noisy; a Poisson noise floor around the forecast is already about half the observed error. The value is in the calibrated distribution, not in a better mean.

**6. The quantile model's median has 9% lower WAPE than MA-28. Isn't that a point-forecast win?**
Not on its own. The median minimises absolute error and WAPE is an absolute-error metric, while MA-28 and the XGBoost mean estimate the mean. Part of the gap is a difference of objective. The like-for-like comparison is mean against mean, and that is on par.

**7. You report inventory savings against B3a. Do they prove the forecasts are better?**
No, and I tested it. In a shuffled-demand null test I replaced realised demand with demand unrelated to the forecasts and re-ran the simulator: the matched-inventory reduction was still positive (7.5% and 27.0% under two nulls on the test window, against 16.0% with real demand). Two policies with different order-up-to shapes need different inventory at equal fill even against random demand. So I report the inventory comparison as descriptive and rely on pinball loss and coverage as the skill evidence. My own expectation that the null would tie was wrong; I logged it.

**8. What is the touch log and pre-registration?**
Before scoring anything I committed the metrics, decision rules and expected outcomes to a decision log; every time test-window data was used, a row was written into a touch log before the run. Choices were made on four tuning folds only. It stops me changing the question after seeing the answer, and it lets a reader see exactly what the test window was used for.

**9. How did you avoid leakage?**
Rolling-origin folds with an 84-day embargo between the fit set and the cutoff, model versions refit only on data before their cutoffs, a written leakage note for every feature, tests that perturb future sales and prices and check no input changes, and a guard that refuses any tuning-stage target window reaching the test window.

**10. What went wrong along the way?**
Several things, all logged: I reported a quantile-crossing rate of 0.000 that had measured already-sorted output (the real rate is 2.4%); an early README claim described the raw-target XGBoost while the pipeline used the normalised one; my first hand-recomputation flagged disagreements that were errors in the check (float32 tolerance, closed days); a clean-clone run exposed a broken Phase 5 script; and integration testing against Postgres found a crash on database restart and an over-strict id validator.

**11. Sales are not demand. How do you handle stock-outs?**
I don't filter them; the forecasts are of sales, and I say so. About 11% of high-velocity days (34% of low-velocity days) sit inside zero runs of 14+ days that look like stock-outs, delistings or seasonal absence. A hindsight diagnostic drops zero-demand cycles from every policy's metrics; it never serves as a headline.

**12. Why pinball loss, and why a "discrete-aware" coverage test?**
Pinball loss is the proper scoring rule for a quantile. Demand is integer and the policy orders `ceil(q)`, so the right question is whether demand is at or below the ordered quantity; with ties (large for slow movers) coverage can sit above the nominal level without the model being wrong, so the test compares nominal against both the coverage and the strictly-below coverage.

**13. How does the policy use the forecast?**
Order-up-to level S = ceil(q at the service level); order max(0, S - inventory position). The stockout-risk label is a band from the same quantiles (HIGH below the median, MEDIUM up to P90, LOW above, an overstock flag above P99), explicitly not a probability; a label investigation showed it over-warns for medians below one unit and inside long zero runs.

**14. Why is the public demo precomputed, with no live model?**
Cost and reliability: free hosting tiers sleep, Hugging Face Docker Spaces need a paid plan, and a five-minute health check keeps one free service awake but not two. Serving precomputed results removes the model from the public path. The FastAPI service stays in the repository with hash-verified model files for local use. The demo is built and tested locally; it is not deployed yet.

**15. How do you know the results reproduce?**
Partly. A clean clone reproduces the data pipeline (features exactly equal) and the tests. The fitting stages from Phase 6 onward were not reproduced from cold; the results rest on frozen model files whose recorded SHA-256 hashes I re-verified (34 of 34), and one cheap version was retrained from scratch and compared within a stated tolerance ([`results/phase11_retrain_check.md`](results/phase11_retrain_check.md)).

**16. What would you do next?**
As a separately pre-registered v2 reported next to v1: drop the series-age input (a post-hoc ablation already lowered tuning loss by 1.7%), train on many more of the M5 series, try a Tweedie or negative-binomial loss and search wider; and replace the inventory statistic by a paired comparison that ties under the null.

**17. What are the main limits?**
One test window and 300 series from one retailer; post-hoc benchmark; sales instead of demand; simulated costs and lead times; horizon 7 has no test-window metrics; see [`limitations.md`](limitations.md).
