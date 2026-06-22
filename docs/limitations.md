# Limitations

What this project does not show, and what would change a conclusion. Numbers cited here come from logged runs in [`results/`](results/).

## What the evidence supports
- On the held-out test window the quantile forecasts have lower scaled pinball loss than the post-hoc empirical-residual benchmark (horizon 10: 0.2296 vs 0.2706, difference -0.0410, 95% interval [-0.0504, -0.0337]; horizon 14: 0.2444 vs 0.2985, [-0.0642, -0.0460]) and coverage of the ordered quantity consistent with nominal at 0.80 to 0.99.
- Nothing else in the repository is offered as skill evidence.

## What it does not show
1. **The mean forecast is not better than a 28-day average.** Test-window WAPE 0.3436 vs 0.3504 (horizon 10) and 0.3146 vs 0.3199 (horizon 14): on par. In the tuning folds it was 4% to 5% worse. The median forecast's lower WAPE (0.3189 at horizon 10) is partly an objective effect: the median minimises absolute error and WAPE is an absolute-error metric.
2. **The inventory comparisons are descriptive, not skill evidence.** A shuffled-demand null test gives a positive matched-inventory reduction against B3a even when realised demand is unrelated to the forecasts (test window: 7.5% and 27.0% under the two nulls, against 16.0% with real demand; tuning folds: 9.7%, 13.0%, 3.9%). The statistic mixes forecast skill with the different shapes of the two rules' order-up-to levels.
3. **The benchmark that matters is post-hoc.** B3a and B3c were defined after the tuning results were known. Against the pre-specified normal-sigma policy the gap is larger, but mostly because of the normal-sigma assumption, which failed its own constant-variance check for high-velocity series.
4. **One test period, one retailer, 300 series.** The bootstrap resamples items, not periods; it says nothing about other seasons, other stores or other categories. The tuning folds were used both to select and to report, so tuning-fold numbers are mildly optimistic.
5. **Horizon 7 has no test-window metrics.** Only horizons 10 and 14 were replayed on the test window.
6. **Sales are not demand.** History is censored when stock ran out; about 11% of high-velocity days (15% mid, 34% low) sit inside zero runs of 14 or more days that look like stock-outs, delistings or seasonal absence. The data is not filtered; the forecasts are of sales.
7. **The replay's assumptions are illustrative.** No stock, lead-time or cost data exist; costs are 70% of price with 25% a year holding; planners see true demand history even when the simulated stock ran out; lead times are fixed (or Uniform{2, 3, 4} in one sensitivity); one Sunday review for every series. The cost sweep is a sensitivity: the other policies' minima sit at their 0.80 grid edge and cannot be extended, so the comparison of best cases is unresolved.
8. **The four-point curves are coarse.** The matched-inventory interpolation rests on four points per policy; the two anchorings differ, and moderate-level reductions (3.5% to 9.5%) were not bootstrapped one setting at a time.
9. **Explanations are associations.** No promotion column exists (price is compared with the usual price); series age carries weight in the model and is not interpreted (a post-hoc ablation dropping it lowered tuning-fold loss by 1.7%, left as future work).
10. **Reproducibility is partial.** The data pipeline and tests reproduce from a clean clone. The fitting stages from Phase 6 onward were not reproduced from cold; the results rest on frozen model files with recorded, re-verified hashes, and one retrained version was compared with them within a stated tolerance ([retrain check](results/phase11_retrain_check.md)).
11. **The demo is not deployed.** It is built and tested locally; no monitor exists; the per-series quantile tables are held back until the data-use terms are confirmed.

## What would change these conclusions
A second test period or other stores; the full M5 item set; a pre-registered v2 (dropping series age, more series, a Tweedie or negative-binomial loss, a wider search) reported next to v1; real stock, lead-time and cost data; a paired comparison built so that it ties under the null.
