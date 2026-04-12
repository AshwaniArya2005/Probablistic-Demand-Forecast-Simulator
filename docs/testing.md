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
| Learned models, synthetic truth (6-7) | `test_learned_synthetic.py`: linear truth recovered, step learned by nonlinear models, all-zero target, quantiles near exact Poisson quantiles with coverage bounded by the exact CDF, quantiles sorted. Runs on two reference learners now; **Linear Regression, Random Forest, XGBoost and quantile XGBoost must register in `models.REGISTRY` with `learned=True` (and `nonlinear=True` for trees) and pass the same tests** | harness done and validated; awaiting the models |
| Conformal (8) | `test_conformal.py` | **pending, see below** |
| SHAP (9) | `test_shap.py` | **pending, see below** |
| Simulator and policies (10) | `test_simulator.py` | **pending, see below** |

## Obligations for components not built yet

### conformal
- Coverage on exchangeable synthetic data: split-conformal upper bounds cover between `alpha` and `alpha + 1/(n+1)` (finite-sample guarantee), checked over many seeded repeats.
- Hand-computed offset: scores 1..10, alpha = 0.8 gives the `ceil((n+1) alpha) = 9`th smallest score = 9.
- Guard: with too few scores for the requested level (a 0.99 offset needs at least 99), the code refuses or falls back explicitly and says so; it never silently returns the sample maximum.
- Pooling by velocity segment, scores divided by series scale and offsets multiplied back by the series' own scale (design section 5); the scale multiplies back exactly (hand example).
- No calibration-window score may come from a fit-set target day (reuses `versions.calibration_mask`).

### shap
- Additivity: for every explained row, base value + sum of SHAP values equals the model output (to numerical tolerance), for the mean model and each explained quantile model.
- Wording: generated explanation text says "price is X% below usual" and never contains "promotion" or "promo"; X is computed from `price_rel_now` (hand example: 0.85 gives "15% below usual").
- Deterministic output for a fixed model and row; features named as in `docs/features.md`.

### simulator
- **Hand-computed scenario:** one series, R = 7, L = 3, known demand list and order-up-to level S: the expected on-hand path, orders, receipts, lost sales, fill rate and average inventory are written out by hand in the test.
- **Timing:** an order placed after close of day t arrives before opening of day t + L + 1; the protection interval is exactly days t+1..t+R+L (matches the target definition).
- **Conservation invariants** (property tests over seeded random demand): start stock + receipts - sales = end stock; demand = sales + lost sales; nothing received that was not ordered `L` days earlier; inventory never negative under lost sales.
- **Limit cases:** S = 0 gives fill rate 0; a very large S gives fill rate 1 and no stockouts; zero demand gives zero sales and no stockouts; closed days replay zero demand while holding cost still accrues; naive rule with c = 1 equals a hand-computed `ceil(mu28 * P)`.
- **Leakage:** perturbing demand after review date t leaves every decision (S and order quantity) at or before t unchanged; policies read historical M5 sales, never simulated sales.
- Determinism for a fixed seed; the version schedule of `versions.py` selects the model used at each review date.
