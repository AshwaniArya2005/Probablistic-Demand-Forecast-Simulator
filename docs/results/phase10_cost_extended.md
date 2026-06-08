# Phase 10 cost sweep with the naive grid extended (post-hoc)

Run 2026-09-21 at code commit `14a450d`. **Post-hoc**: made after the primary run showed the naive rule's rho 4 minimum at
the lower edge of its grid (c = 1.00). The naive grid is extended downward by 0.25 (floor 0.25, the pre-registered floor); nothing else changes and the base-case numbers reproduce the primary run.
Test-window data, logged in the touch log as its own entry.

## Minimum total cost per policy, per rho (currency units; a minimum at either end of a grid is flagged)

| policy                          | rho 4 (critical ratio 0.80)     | rho 9 (critical ratio 0.90)     | rho 19 (critical ratio 0.95)    | rho 99 (critical ratio 0.99)   |
|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------|
| B3a                             | 1,281 at 0.8 (edge of its grid) | 1,555 at 0.8 (edge of its grid) | 2,069 at 0.9                    | 3,819 at 0.95                  |
| B2 / B2-sqrt                    | 2,293 at 0.8 (edge of its grid) | 2,413 at 0.8 (edge of its grid) | 2,652 at 0.8 (edge of its grid) | 4,475 at 0.9                   |
| Quantile                        | 1,213 at 0.8 (edge of its grid) | 1,503 at 0.8 (edge of its grid) | 1,931 at 0.9                    | 3,617 at 0.95                  |
| Naive, primary grid             | 1,174 at 1.0 (edge of its grid) | 1,604 at 1.25                   | 2,146 at 1.5                    | 4,223 at 2.25                  |
| Naive, extended grid (post-hoc) | 1,174 at 1.0                    | 1,604 at 1.25                   | 2,146 at 1.5                    | 4,223 at 2.25                  |

## Naive rule: total cost by c (c up to 3.00)

|    c | rho 4   | rho 9   | rho 19   | rho 99   |
|-----:|:--------|:--------|:---------|:---------|
| 0.25 | 2,392   | 5,248   | 10,960   | 56,659   |
| 0.5  | 1,721   | 3,550   | 7,207    | 36,470   |
| 0.75 | 1,330   | 2,434   | 4,641    | 22,300   |
| 1    | 1,174   | 1,793   | 3,031    | 12,933   |
| 1.25 | 1,252   | 1,604   | 2,307    | 7,932    |
| 1.5  | 1,468   | 1,694   | 2,146    | 5,761    |
| 1.75 | 1,752   | 1,912   | 2,233    | 4,800    |
| 2    | 2,070   | 2,192   | 2,435    | 4,382    |
| 2.25 | 2,418   | 2,513   | 2,703    | 4,223    |
| 2.5  | 2,769   | 2,848   | 3,007    | 4,276    |
| 2.75 | 3,118   | 3,187   | 3,327    | 4,446    |
| 3    | 3,488   | 3,548   | 3,668    | 4,625    |

The target-driven policies' minima at alpha 0.80 cannot be extended (no quantile models between 0.50 and 0.80), so those results stay **unresolved**.
