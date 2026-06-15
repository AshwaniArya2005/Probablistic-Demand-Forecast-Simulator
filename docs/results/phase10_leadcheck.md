# Random lead time at alpha 0.99: fill rates printed (test window, base case, quantile policy)

Output of `ml/lead_check.py` (touch-log row "Phase 10 diagnostic leadcheck"), run 2026-09-21 at the commit that added it. Fixed lead time 3 against ten independent random-lead draws (Uniform{2, 3, 4}, seeds 0 to 9, mean of
the draws shown). Aggregates only.

| alpha | lead time | fill rate | lost units | units demanded | cycle service |
|---:|---|---:|---:|---:|---:|
| 0.80 | fixed 3 | 0.931842 | 4931 | 72347 | 0.884800 |
| 0.80 | random x10 (mean) | 0.929457 | 5103.6 | 72347 | 0.866200 |
| 0.90 | fixed 3 | 0.961090 | 2815 | 72347 | 0.940533 |
| 0.90 | random x10 (mean) | 0.959881 | 2902.5 | 72347 | 0.931453 |
| 0.95 | fixed 3 | 0.976765 | 1681 | 72347 | 0.970667 |
| 0.95 | random x10 (mean) | 0.975916 | 1742.4 | 72347 | 0.964240 |
| 0.99 | fixed 3 | 0.995757 | 307 | 72347 | 0.997067 |
| 0.99 | random x10 (mean) | 0.995761 | 306.7 | 72347 | 0.996547 |

Per-draw fill rate range at 0.99: 0.995148 to 0.996227 (0.80: 0.927585 to 0.931220; 0.90: 0.958810 to 0.961408; 0.95: 0.975244 to 0.977221).

**Draws applied:** in draw 0 the lead times are 2 / 3 / 4 in 33.4% / 33.3% / 33.3% of the (review, series) cells (mean 2.999); of 5,394 orders placed, 3,576 had a realised lead other than 3; 3,754 of 581,100
series-day end-of-day on-hand values differ from the fixed-lead replay, and order quantities differ in 18 of 8,700 cells (order-up-to logic absorbs most timing differences).

**Reading:** at 0.99 the mean fill rate under random lead time (0.995761) equals the fixed-lead value (0.995757) to the fifth decimal, and the single-draw spread (0.995148 to 0.996227) is more than a hundred times
the difference, so the fill rate at 0.99 is not measurably affected; cycle service is (0.99707 to 0.99655). None of the 7,500 series-cycles had an order-up-to level of 0, so the 307 unmet units are demand beyond the 0.99
quantile, not a structural floor. This is not a bug and not saturation.
