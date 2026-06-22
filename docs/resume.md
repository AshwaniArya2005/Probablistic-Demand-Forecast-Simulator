# Resume bullets and claims to avoid

Numbers below come only from the test-window forecast metrics ([`results/phase10_test_forecast_metrics.md`](results/phase10_test_forecast_metrics.md)): held-out 26-week window, 25 review dates x 300 series, mean over model versions v1 to v4. Always state the horizon.

## Bullets (pick two or three; adjust length to the page)

- Built a probabilistic demand forecaster (XGBoost multi-quantile, P10 to P99) for 300 M5 retail item-store series with rolling-origin validation, embargoed retraining and leakage tests. On a held-out 26-week test window, its **10-day** quantile forecasts had **15% lower scaled pinball loss** than a post-hoc empirical-residual benchmark built on the same point forecast (0.2296 vs 0.2706; item-bootstrap 95% interval for the difference [-0.0504, -0.0337]; lower in 4 of 4 model versions), with coverage of the ordered quantity consistent with nominal at all four service levels (0.80 to 0.99). The **14-day** horizon gave 0.2444 vs 0.2985 ([-0.0642, -0.0460]).
- Tested and reported an unflattering result honestly: the pipeline's **mean** forecast was only on par with a 28-day moving average on the test window (WAPE 0.344 vs 0.350 at 10 days), so the contribution is the calibrated distribution, not point accuracy.
- Ran evaluation like a pre-registered study: rules committed before scoring, a test-window touch log written before every use, predictions recorded and reported whether they held or failed, and a shuffled-demand null test that showed my inventory-savings statistic was not evidence of skill, after which I re-scoped the claims.
- Built a discrete-event inventory replay (order-up-to, lost sales, integer units) and verified it against an independent hand-coded implementation, a conservation-invariant test suite and a clean-clone reproduction that caught a real defect.
- Designed a results demo (static frontend, Express API, Postgres) that serves precomputed results only, with cheap and deep health endpoints, rate limiting, generic error bodies, a snapshot fallback for cold starts and a publication gate for data-derived tables; built and tested locally, with a hash-verified FastAPI model service kept out of the public path.

## One-line versions
- "Calibrated 10-day demand quantiles for 300 retail series: 15% lower pinball loss than an empirical-residual benchmark on a held-out window (95% CI on the difference excludes zero; horizon 10)."
- "Pre-registered evaluation with a test-window touch log; a null test caught an over-claim before I made it."

## Do not claim
- Any inventory saving as evidence of forecast skill (the null test shows the comparison is positive without any skill), and no inventory percentages on a resume.
- That the mean forecast beats MA-28, or quote the median-forecast WAPE gap (9% at 10 days) without saying the median minimises absolute error.
- The 36% or 39% pinball gap against the normal-sigma policy on its own (mostly the normal-sigma assumption).
- Tuning-fold numbers as results, horizon 7 test-window numbers (none exist), production deployment, uptime or monitoring (none exists yet), or anything about the M5 data-use terms until they are confirmed.
