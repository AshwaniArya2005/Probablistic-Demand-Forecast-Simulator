# Testing

**A phase is complete only when its tests are written, pass in the fast suite, and (where marked) pass on the real data locally.**
This applies to every phase from Phase 5 on. A skipped placeholder for a phase's component blocks that phase.

```
uv run pytest                 # fast suite: synthetic fixtures only, needs no data; this is what CI runs
uv run pytest -m realdata     # real-data checks: need data/processed built by ml/prepare.py and ml/features.py
```

## Declaring a phase done

1. Commit the phase's code and tests. Replace any placeholder tagged `# PENDING-PHASE: N` for this phase with real tests.
2. `uv run python ml/record_tests.py N` runs the fast suite and the real-data tests on the clean tree and appends the result to `docs/results/test_log.md` (exit code 1 if anything failed).
3. Add N to `docs/phase_status.json` and commit. `tests/test_phase_gate.py` then fails if a placeholder for N remains, if a model N must deliver is not in `models.REGISTRY` (`ml/phase_gate.py: REQUIRED_MODELS`), or if the log has no passing fast + real-data row for N.

Rules for every test:
- **No unmeasured accuracy numbers.** A test asserts definitions, invariants, hand-computed values and properties of synthetic problems with known truth. It never asserts "WAPE < x" or "service level > y" for the project's data.
- **Small seeded synthetic fixtures** (`tests/conftest.py`: five series incl. a stock-out run, a dead series and a short series). Exact pmf arithmetic is preferred to simulation where a test needs a true quantile.
- **A check that can pass can also fail.** Where a suite could pass vacuously, a deliberately broken model or data is fed to it and must be rejected (`test_causality_check_rejects_broken_models`, `test_the_suite_rejects_broken_models`, "blind test" assertions in `test_features.py`).
- Real-data tests carry `@pytest.mark.realdata`, are excluded by default and by CI, and are run before a phase is closed.

## Status

| Component (phase) | Tests | Status |
|---|---|---|
| Features (4) | `test_features.py`: brute-force reference, sales/price perturbation leakage tests, structure; `-m realdata` repeats them on the real panel and checks the calendar against `calendar.csv` | done |
| Model contract (5) | `test_models_contract.py`, parametrized over `models.REGISTRY`: shape, finite, non-negative, seeded determinism, save/load round trip, causality (scrambled non-inputs, row order, other rows), dead and short series, input not mutated, sorted quantiles for quantile models | done for the 3 baselines; any registered model is covered automatically |
| Baselines (5) | `test_baselines.py`: naive, MA-28, seasonal naive against hand-computed values, fallback and error rules | done |
| Metrics (5) | `test_metrics.py`: WAPE, MAE, RMSE, MAPE, MASE (zero and missing scale excluded), naive scale, pinball (exactly minimised at the true quantile, from the Poisson pmf), scaled pinball, interval and one-sided coverage vs exact CDF | done |
| Quantile utilities (5) | `test_quantiles.py`: sorting fixes crossing and never raises pinball loss; quantile of a P-day sum is not the sum of daily quantiles (Poisson(1) x 7: 14 vs 10) | done |
| Fit-set embargo, calibration start, version schedule (5) | `test_versions.py`: for every version and P, fit targets end by `cutoff - 84 days`, calibration window starts exactly where the fit set ends and its targets are observed by the cutoff, use windows partition the 29 review dates, staleness is 12-20 weeks, version 1 fit end equals the tuning cutoff; a no-embargo control fails; `-m realdata` on the real table | done |
| Tuning folds and test-window guard (5) | `test_folds.py`: 12 Sundays each, ordered, last P = 14 target ends on the tuning cutoff, one fold covers Nov-Jan, `assert_tuning_only` rejects test-window dates | done |
| Learned point models (6) | `test_learned_synthetic.py` runs linear truth, step learning (trees), all-zero target on **Linear Regression, Random Forest and XGBoost** (registered in `models.REGISTRY`, so `test_models_contract.py` covers them too: shape, no NaN or negatives, seeded determinism, save/load, causality, dead and short series). `test_learned_models.py`: scale = `max(mean_28, floor) * P` by hand, normalised forecasts multiply the scale back, LR clips at zero and ignores unseen ids, early-stopping split is temporal with an embargo (hand dates), RF row cap and depth cap bind, ablation `drop` removes a column completely. The determinism test found RF's parallel predict was not bit-reproducible; fixed | done |
| Selection rules (6) | `test_selection.py`: simplest-within-0.002, floor tie to 1/28, year-ago gain 0.003, id-arm rule, all on hand-picked numbers with boundary cases | done |
| Quantile models (7) | `xgb_q` (registered, `REQUIRED_MODELS[7]`) passes the contract tests and the synthetic quantile checks (near-true quantiles, coverage bounded by the exact CDF, sorted); `test_learned_models.py`: alphas, sorted/non-negative output, median = `predict`, normalised scale round trip, untrained level rejected, thread count | done |
| Phase 7 pure functions (7) | `test_selection.py` (relative tolerances, the normalisation rule incl. the mean-vs-median disagreement staying raw), `test_metrics.py` (`cov_lo` / `cov_hi` / status bands, interval coverage), `test_quantiles.py` (z values, pooled sigma, benchmark quantile by hand); `test_features.py` covers `ev_named` and `xs_zero` (brute force and leak tests, synthetic and real data); `test_phase7_runner.py` (realdata) checks the feature sets by variant and arm | done |
| Conformal (7) | `test_conformal.py`: hand offsets, `n_min` 4 / 9 / 19 / 99, the guard raises below it and allows the minimum, thin-sample thresholds, segment pooling, scale round trip, clip + re-sort, exact finite-sample coverage on exchangeable data, calibration origins are Sundays from the calibration window only | done |
| SHAP explanations (8) | `test_shap.py` (18 tests): every input in exactly one theme and theme sums equal the total; ratio-to-unit conversion by hand and end to end; sliced boosters reproduce the sorted joint prediction; additivity per quantile in ratio and unit space including rows where quantiles cross (blind-tested fixture); a test that documents the XGBoost 3.4.1 `pred_contribs` defect; sign-to-word agreement (property test); price wording by hand and the absolute price never described as below usual; banned words never appear; subjects agree in number; Sunday-only and tuning-only guards, the test-window flag needs a touch-log reference; determinism and save/load | done |
| Simulator and policies (10) | `test_simulator.py` | **pending, see below** |

## Obligations for components not built yet

### conformal
Pre-registered in design.md section 12 (Phase 7 pre-registration, item 6). Tests to write, replacing the placeholder:
- Coverage on exchangeable synthetic data: split-conformal upper bounds cover between `alpha` and `alpha + 1/(n+1)` (finite-sample guarantee), checked over many seeded repeats.
- Hand-computed offset: scores 1..10, alpha = 0.8 gives `k = ceil(11 * 0.8) = 9`, the 9th smallest score, 9.
- **Too-few-scores guard:** `n_min(alpha) = ceil(alpha / (1 - alpha))` is 4, 9, 19, 99 for 0.80, 0.90, 0.95, 0.99. With `n = n_min - 1` the code raises `InsufficientScores`; at `n = n_min` it returns the maximum (k = n) by rule, not by fallback. It never returns the sample maximum for a smaller n. The thin-sample warning fires below `5 / (1 - alpha)` and is not an error.
- Pooling by the given velocity segment labels (never recomputed, never per series alone); scores divided by the series' scale and the offset multiplied back by the series' own scale (hand example, including the floor).
- Calibration origins are the **Sunday** origins from `versions.calibration_mask`: none from a fit-set target day, none with a target unobserved at the cutoff, no daily origins.
- After adding offsets the quantiles are clipped at 0 and re-sorted across alphas (crossing case by hand).

### shap
Pre-registered in design.md section 12 (Phase 8 pre-registration). Tests to write, replacing the placeholder:
- **Additivity per quantile:** bias + sum of feature contributions equals the raw model output for the P50 and for the service-level quantile, in ratio space and in unit space.
- **Unit conversion:** ratio contributions times the row scale equal unit contributions (hand example: 0.5 x scale 8 = 4.0 units), and end to end against the model.
- **Sign-to-word agreement:** the verb is "raises" iff the theme's unit contribution is positive and "lowers" iff negative; never the other way (property test over random contributions).
- **Wording rules:** price 0.85 gives "price is 15% below usual", 1.10 gives "10% above usual", 1.004 gives "about usual"; none of promotion, promo, discount, on sale, because, caused ever appears.
- **Themes:** every model input is in exactly one theme; theme contributions sum to the total. The absolute `price` is its own theme ("price level") and is never described as below or above usual; only `price_rel_now` and the planned window price features carry that wording.
- **Exactness under crossing:** contributions come from single-target boosters sliced from the model (XGBoost 3.4.1's joint `pred_contribs` and the `shap` package do not reconcile with `predict`), taken per reported rank, and additivity is tested on rows where quantiles cross.
- **Guards:** only Sunday origins are explained; origins whose target window passes the tuning cutoff are refused unless the explicit test-window flag is passed.
- Deterministic output for a fixed model and row.

### simulator
- **Hand-computed scenario:** one series, R = 7, L = 3, known demand list and order-up-to level S: the expected on-hand path, orders, receipts, lost sales, fill rate and average inventory are written out by hand in the test.
- **Timing:** an order placed after close of day t arrives before opening of day t + L + 1; the protection interval is exactly days t+1..t+R+L (matches the target definition).
- **Conservation invariants** (property tests over seeded random demand): start stock + receipts - sales = end stock; demand = sales + lost sales; nothing received that was not ordered `L` days earlier; inventory never negative under lost sales.
- **Limit cases:** S = 0 gives fill rate 0; a very large S gives fill rate 1 and no stockouts; zero demand gives zero sales and no stockouts; closed days replay zero demand while holding cost still accrues; naive rule with c = 1 equals a hand-computed `ceil(mu28 * P)`.
- **Leakage:** perturbing demand after review date t leaves every decision (S and order quantity) at or before t unchanged; policies read historical M5 sales, never simulated sales.
- Determinism for a fixed seed; the version schedule of `versions.py` selects the model used at each review date.
