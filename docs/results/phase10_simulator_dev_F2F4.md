# Phase 10 simulator, development run on F2 and F4

Run 2026-09-21 at code commit `48e2152`. Base case L = 3, R = 7 (P = 10); 2 run(s), 2 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
Development on **F2 (holiday fold) and F4 (summer fold) only**, tuning data (design.md, Phase 10 definitions, item 1); warm-up = the four Sundays before each fold's first origin. For F2 the peak sub-period is the analogue of the holiday peak one year earlier (cycles starting on or before 2015-01-04); F4 counts as rest. **Not the primary result and not a test of the pre-registered predictions** (those are evaluated on the test window only).
**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.


## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 9.7%        | 13.7%        | 43.1%        | 3 of 4             | 22.2%               | [8.2%, 24.3%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | n/a          | n/a          | 0 of 4             | n/a                 | [16.5%, 50.4%]         | 47%                      |
| Naive (c x mu28 x P), widened grid        | 8.9%        | 14.2%       | 18.0%        | 64.8%        | 4 of 4             | 26.5%               | [9.5%, 29.6%]          | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 3.3%        | 6.6%        | 1.9%         | n/a          | 3 of 4             | 3.9%                | [0.1%, 17.2%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 16.2%       | 22.4%       | 33.1%        | 48.6%        | 4 of 4             | 30.1%               | [8.4%, 41.4%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 15 of 23           | 29.2%               | [6.5%, 46.5%]          | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 8.8%        | 15.1%        | 21.8%        | 3 of 4             | 15.3%               | [4.7%, 22.4%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 0.4%        | 4.5%        | 5.0%         | -1.2%        | 4 of 4             | 2.2%                | [-5.3%, 8.1%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 20 of 23           | 52.2%               | [34.5%, 60.4%]         | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 3.6%        | 7.5%        | -3.2%        | 43.0%        | 4 of 4             | 12.7%               | [-2.8%, 18.8%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 7.2%        | 12.9%       | 7.2%         | 8.3%         | 4 of 4             | 8.9%                | [-4.4%, 17.9%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 45.2%               | [39.0%, 52.0%]         | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 3.7%        | 6.2%        | -0.8%        | n/a          | 3 of 4             | 3.0%                | [-5.3%, 20.3%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 3.3%        | n/a         | n/a          | n/a          | 1 of 4             | 3.3%                | [-4.0%, 38.8%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 10 of 23           | 9.1%                | [0.0%, 44.7%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.087 |       0.051 |        0.029 |        0.007 |
| all      | B2 / B2-sqrt |       0.159 |       0.082 |        0.037 |        0.003 |
| all      | Quantile     |       0.086 |       0.044 |        0.02  |        0.008 |
| peak     | B3a          |       0.089 |       0.053 |        0.031 |        0.007 |
| peak     | B2 / B2-sqrt |       0.159 |       0.081 |        0.036 |        0.003 |
| peak     | Quantile     |       0.084 |       0.044 |        0.02  |        0.009 |
| rest     | B3a          |       0.08  |       0.044 |        0.023 |        0.008 |
| rest     | B2 / B2-sqrt |       0.161 |       0.083 |        0.04  |        0.006 |
| rest     | Quantile     |       0.092 |       0.043 |        0.019 |        0.007 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 2.2%        | 5.5%        | -4.7%        | 44.2%        | 4 of 4             | 11.8%               | [4.6%, 20.5%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 22.8%       | 24.4%       | 34.1%        | 48.0%        | 4 of 4             | 32.3%               | [24.4%, 42.2%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 42.8%               | [37.9%, 48.0%]         | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 8.1%        | 13.2%        | n/a          | 2 of 4             | 10.7%               | [-2.3%, 24.6%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 13.8%       | n/a         | n/a          | n/a          | 1 of 4             | 13.8%               | [0.0%, 47.5%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 8 of 23            | 10.0%               | [1.1%, 47.8%]          | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9261 |          0.8867 |                                10.66 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9588 |          0.951  |                                14.8  |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9758 |          0.9792 |                                20.12 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9957 |          0.9974 |                                65.5  |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.98   |          0.959  |                                27.81 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.9888 |          0.9819 |                                39.63 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9917 |          0.9872 |                                49.6  |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9942 |          0.9935 |                                68.68 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9195 |          0.8858 |                                 9.74 |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9527 |          0.9439 |                                12.67 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9716 |          0.9701 |                                16.26 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.9943 |          0.9982 |                                35.39 |
| Naive (c x mu28 x P)                                              | c 1.00     |      0.8471 |          0.7575 |                                 7.01 |
| Naive (c x mu28 x P)                                              | c 1.25     |      0.9114 |          0.8569 |                                 9.89 |
| Naive (c x mu28 x P)                                              | c 1.50     |      0.9441 |          0.9122 |                                13.1  |
| Naive (c x mu28 x P)                                              | c 1.75     |      0.9614 |          0.9431 |                                16.48 |
| Naive (c x mu28 x P)                                              | c 2.00     |      0.9722 |          0.9631 |                                20.01 |
| Naive (c x mu28 x P)                                              | c 2.25     |      0.9793 |          0.9732 |                                23.67 |
| Naive (c x mu28 x P)                                              | c 2.50     |      0.9845 |          0.9797 |                                27.31 |
| Naive (c x mu28 x P)                                              | c 2.75     |      0.9867 |          0.9821 |                                30.94 |
| Naive (c x mu28 x P)                                              | c 3.00     |      0.9888 |          0.9879 |                                34.73 |
| Naive (c x mu28 x P)                                              | c 3.50     |      0.9906 |          0.9906 |                                42.23 |
| Naive (c x mu28 x P)                                              | c 4.00     |      0.992  |          0.9918 |                                49.91 |
| Naive (c x mu28 x P)                                              | c 4.50     |      0.9928 |          0.9924 |                                57.71 |
| Naive (c x mu28 x P)                                              | c 5.00     |      0.9933 |          0.9928 |                                65.52 |
| Naive (c x mu28 x P)                                              | c 5.50     |      0.9937 |          0.9931 |                                73.41 |
| Naive (c x mu28 x P)                                              | c 6.00     |      0.994  |          0.9938 |                                81.31 |
| Naive (c x mu28 x P)                                              | c 6.50     |      0.9941 |          0.9938 |                                89.36 |
| Naive (c x mu28 x P)                                              | c 7.00     |      0.9942 |          0.9939 |                                97.25 |
| Naive (c x mu28 x P)                                              | c 7.50     |      0.9944 |          0.994  |                               105.56 |
| Naive (c x mu28 x P)                                              | c 8.00     |      0.9945 |          0.994  |                               113.68 |
| Naive (c x mu28 x P)                                              | c 8.50     |      0.9947 |          0.9944 |                               121.95 |
| Naive (c x mu28 x P)                                              | c 9.00     |      0.9947 |          0.9944 |                               130.15 |
| Naive (c x mu28 x P)                                              | c 9.50     |      0.9948 |          0.9946 |                               138.43 |
| Naive (c x mu28 x P)                                              | c 10.00    |      0.9949 |          0.9949 |                               146.67 |

Naive c-grid widening (rule 3): start [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]; top target (highest fill among the other policies at alpha 0.99) 0.9957; bottom target (lowest at alpha 0.80) 0.9195.
- up: c = 3.50 added (max naive fill 0.9906 vs top 0.9957)
- up: c = 4.00 added (max naive fill 0.9920 vs top 0.9957)
- up: c = 4.50 added (max naive fill 0.9928 vs top 0.9957)
- up: c = 5.00 added (max naive fill 0.9933 vs top 0.9957)
- up: c = 5.50 added (max naive fill 0.9937 vs top 0.9957)
- up: c = 6.00 added (max naive fill 0.9940 vs top 0.9957)
- up: c = 6.50 added (max naive fill 0.9941 vs top 0.9957)
- up: c = 7.00 added (max naive fill 0.9942 vs top 0.9957)
- up: c = 7.50 added (max naive fill 0.9944 vs top 0.9957)
- up: c = 8.00 added (max naive fill 0.9945 vs top 0.9957)
- up: c = 8.50 added (max naive fill 0.9947 vs top 0.9957)
- up: c = 9.00 added (max naive fill 0.9947 vs top 0.9957)
- up: c = 9.50 added (max naive fill 0.9948 vs top 0.9957)
- up: c = 10.00 added (max naive fill 0.9949 vs top 0.9957)
Final grid: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

## 6. Sigma form used by the point policy (Phase 9 check, per model unit and segment)

| segment   |   B2 (constant CV) |   B2 (constant CV, check failed for both forms) |   B2-sqrt |   model versions |
|:----------|-------------------:|------------------------------------------------:|----------:|-----------------:|
| low       |                  0 |                                               0 |         2 |                2 |
| mid       |                  0 |                                               1 |         1 |                2 |
| high      |                  0 |                                               2 |         0 |                2 |

## 7. Total cost: minimum over each policy's own settings, per stockout-to-holding ratio rho (currency units, pooled; a sensitivity, not the headline)

| policy       | rho 4 (critical ratio 0.80)     | rho 9 (critical ratio 0.90)     | rho 19 (critical ratio 0.95)    | rho 99 (critical ratio 0.99)     |
|:-------------|:--------------------------------|:--------------------------------|:--------------------------------|:---------------------------------|
| B3a          | 1,183 at 0.8 (edge of its grid) | 1,488 at 0.8 (edge of its grid) | 1,993 at 0.9                    | 4,021 at 0.95                    |
| B2 / B2-sqrt | 2,302 at 0.8 (edge of its grid) | 2,395 at 0.8 (edge of its grid) | 2,583 at 0.8 (edge of its grid) | 4,023 at 0.9                     |
| Quantile     | 1,137 at 0.8 (edge of its grid) | 1,466 at 0.8 (edge of its grid) | 1,873 at 0.9                    | 3,795 at 0.99 (edge of its grid) |
| Naive        | 1,113 at 1.0 (edge of its grid) | 1,535 at 1.25                   | 2,066 at 1.5                    | 3,875 at 2.5                     |
