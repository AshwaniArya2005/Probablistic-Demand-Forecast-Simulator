# Phase 10 simulator, primary test-window run

Run 2026-09-21 at code commit `1dec014`. Base case L = 3, R = 7 (P = 10); 1 run(s), 5 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
**The primary test-window run** (design.md, Phase 10 pre-run rules item 2): base case L = 3, R = 7; four warm-up reviews (25 Oct to 15 Nov) excluded, 25 review dates counted (22 Nov to 8 May); model versions v0 to v4 serve their review dates. Peak = cycles starting on or before 2016-01-03 (the holiday peak), rest = later cycles.
**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.

## 0. Pre-registered predictions (design.md section 12, Phase 10 definitions, item 2), evaluated mechanically

**(a) Rest-period under-service.** Cycle service minus target alpha, peak (cycles starting on or before 2016-01-03) vs rest:

| policy                      |   alpha |   gap in peak |   gap in rest | rest below target   | rest gap smaller than peak gap   |
|:----------------------------|--------:|--------------:|--------------:|:--------------------|:---------------------------------|
| quantile policy             |    0.8  |         0.082 |         0.086 | no                  | no                               |
| quantile policy             |    0.9  |         0.032 |         0.043 | no                  | no                               |
| quantile policy             |    0.95 |         0.017 |         0.022 | no                  | no                               |
| quantile policy             |    0.99 |         0.005 |         0.008 | no                  | no                               |
| point policy (B2 / B2-sqrt) |    0.8  |         0.152 |         0.162 | no                  | no                               |
| point policy (B2 / B2-sqrt) |    0.9  |         0.077 |         0.081 | no                  | no                               |
| point policy (B2 / B2-sqrt) |    0.95 |         0.033 |         0.039 | no                  | no                               |
| point policy (B2 / B2-sqrt) |    0.99 |        -0.001 |         0.005 | no                  | no                               |

Quantile policy: rest below target at every alpha and by more than in the peak: **NOT CONFIRMED**. Point policy (relative claim only): the rest gap is smaller than the peak gap at 0 of 4 alphas.

**(b) 0.99 is inefficient for the quantile policy.** Marginal inventory per percentage point of fill rate: 0.90 to 0.95: 2.17 units per series-day; 0.95 to 0.99: 10.93.
Cycle service at 0.99 in the peak: 0.9950 (prediction: at or above 0.99). Verdict: **CONFIRMED** (marginal cost larger: yes; peak service at or above 0.99: yes).

**(c) A smaller edge over B3a than over the point policy** (mean inventory reduction at matched fill rate over the matched settings; the comparator-anchored statistic is the one defined for both comparators):

| scope   | anchor                        | vs B3a   | vs B2 / B2-sqrt   | smaller vs B3a   |
|:--------|:------------------------------|:---------|:------------------|:-----------------|
| pooled  | comparator (defined for both) | 16.0%    | 29.5%             | yes              |
| pooled  | quantile (pre-registered)     | 12.7%    | 31.8%             | yes              |
| high    | comparator (defined for both) | 20.0%    | 26.0%             | yes              |
| high    | quantile (pre-registered)     | 7.9%     | n/a               | n/a              |
| mid     | comparator (defined for both) | 17.7%    | 10.6%             | no               |
| mid     | quantile (pre-registered)     | 15.3%    | 19.9%             | yes              |
| low     | comparator (defined for both) | 4.6%     | 7.6%              | yes              |
| low     | quantile (pre-registered)     | 19.4%    | 13.2%             | no               |

Verdict on the comparator-anchored statistic, pooled and in the high and mid segments: **NOT CONFIRMED**.

## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 9.2%        | 16.2%        | n/a          | 2 of 4             | 12.7%               | [9.5%, 24.4%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | 31.8%        | n/a          | 1 of 4             | 31.8%               | [28.0%, 41.9%]         | 58%                      |
| Naive (c x mu28 x P), widened grid        | 11.7%       | 17.2%       | 25.4%        | n/a          | 3 of 4             | 18.1%               | [14.3%, 32.0%]         | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.5%        | 7.7%        | 5.6%         | 46.3%        | 4 of 4             | 16.0%               | [0.7%, 24.9%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 31.1%       | 20.8%       | 26.4%        | 39.6%        | 4 of 4             | 29.5%               | [21.9%, 35.9%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 45.3%               | [12.6%, 52.4%]         | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 2.9%        | 11.5%       | -0.7%        | n/a          | 3 of 4             | 4.6%                | [-2.8%, 15.6%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 5.0%        | 8.2%        | 11.3%        | 5.8%         | 4 of 4             | 7.6%                | [-1.0%, 12.7%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 20 of 23           | 43.9%               | [12.1%, 50.9%]         | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 6.4%        | 10.0%       | 7.0%         | 47.4%        | 4 of 4             | 17.7%               | [1.1%, 27.2%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 13.2%       | 14.9%       | 2.3%         | 11.9%        | 4 of 4             | 10.6%               | [2.2%, 16.8%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.7%               | [19.3%, 58.0%]         | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 4.9%        | 6.6%         | 48.5%        | 3 of 4             | 20.0%               | [-3.1%, 26.2%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 6.8%        | 19.1%       | 31.0%        | 47.0%        | 4 of 4             | 26.0%               | [7.8%, 36.7%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 15 of 23           | 32.8%               | [0.5%, 54.3%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.089 |       0.049 |        0.028 |        0.007 |
| all      | B2 / B2-sqrt |       0.159 |       0.08  |        0.037 |        0.003 |
| all      | Quantile     |       0.085 |       0.041 |        0.021 |        0.007 |
| peak     | B3a          |       0.092 |       0.047 |        0.027 |        0.006 |
| peak     | B2 / B2-sqrt |       0.152 |       0.077 |        0.033 |       -0.001 |
| peak     | Quantile     |       0.082 |       0.032 |        0.017 |        0.005 |
| rest     | B3a          |       0.088 |       0.049 |        0.028 |        0.008 |
| rest     | B2 / B2-sqrt |       0.162 |       0.081 |        0.039 |        0.005 |
| rest     | Quantile     |       0.086 |       0.043 |        0.022 |        0.008 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 0.0%        | 6.7%        | 7.6%         | n/a          | 3 of 4             | 4.8%                | [-7.2%, 20.5%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 23.4%       | 15.1%       | 19.0%        | 28.2%        | 4 of 4             | 21.4%               | [2.0%, 32.5%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 10 of 23           | 25.3%               | [0.1%, 54.5%]          | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 8.4%        | 5.2%         | 48.2%        | 3 of 4             | 20.6%               | [9.0%, 26.3%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 32.5%       | 21.8%       | 27.9%        | 42.5%        | 4 of 4             | 31.2%               | [23.4%, 38.2%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 45.9%               | [39.7%, 54.1%]         | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9327 |          0.8888 |                                10.89 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9636 |          0.9488 |                                15.25 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9797 |          0.9777 |                                21.3  |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9955 |          0.9972 |                                69.7  |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.9764 |          0.9593 |                                24.46 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.986  |          0.9796 |                                34.16 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9898 |          0.9875 |                                42.35 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9934 |          0.9933 |                                58.03 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9318 |          0.8848 |                                10.31 |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9611 |          0.9405 |                                13.53 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9768 |          0.9707 |                                16.93 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.9958 |          0.9971 |                                37.68 |
| Naive (c x mu28 x P)                                              | c 1.00     |      0.8576 |          0.7447 |                                 7.19 |
| Naive (c x mu28 x P)                                              | c 1.25     |      0.9195 |          0.8547 |                                10.28 |
| Naive (c x mu28 x P)                                              | c 1.50     |      0.9497 |          0.9111 |                                13.69 |
| Naive (c x mu28 x P)                                              | c 1.75     |      0.9652 |          0.9412 |                                17.28 |
| Naive (c x mu28 x P)                                              | c 2.00     |      0.9742 |          0.9599 |                                21.05 |
| Naive (c x mu28 x P)                                              | c 2.25     |      0.9804 |          0.972  |                                24.99 |
| Naive (c x mu28 x P)                                              | c 2.50     |      0.9842 |          0.978  |                                28.9  |
| Naive (c x mu28 x P)                                              | c 2.75     |      0.9865 |          0.9813 |                                32.81 |
| Naive (c x mu28 x P)                                              | c 3.00     |      0.9885 |          0.9856 |                                36.86 |
| Naive (c x mu28 x P)                                              | c 3.50     |      0.9905 |          0.9899 |                                44.9  |
| Naive (c x mu28 x P)                                              | c 4.00     |      0.9916 |          0.9927 |                                53.17 |
| Naive (c x mu28 x P)                                              | c 4.50     |      0.9924 |          0.9939 |                                61.52 |
| Naive (c x mu28 x P)                                              | c 5.00     |      0.993  |          0.9939 |                                69.96 |
| Naive (c x mu28 x P)                                              | c 5.50     |      0.9936 |          0.9939 |                                78.48 |
| Naive (c x mu28 x P)                                              | c 6.00     |      0.9942 |          0.9941 |                                87.05 |
| Naive (c x mu28 x P)                                              | c 6.50     |      0.9945 |          0.9944 |                                95.81 |
| Naive (c x mu28 x P)                                              | c 7.00     |      0.9946 |          0.9944 |                               104.36 |
| Naive (c x mu28 x P)                                              | c 7.50     |      0.9948 |          0.9947 |                               113.38 |
| Naive (c x mu28 x P)                                              | c 8.00     |      0.9949 |          0.9949 |                               122.23 |
| Naive (c x mu28 x P)                                              | c 8.50     |      0.9951 |          0.9953 |                               131.2  |
| Naive (c x mu28 x P)                                              | c 9.00     |      0.9951 |          0.9953 |                               140.1  |
| Naive (c x mu28 x P)                                              | c 9.50     |      0.9952 |          0.9953 |                               149.17 |
| Naive (c x mu28 x P)                                              | c 10.00    |      0.9952 |          0.9953 |                               158.17 |

Naive c-grid widening (rule 3): start [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]; top target (highest fill among the other policies at alpha 0.99) n/a; bottom target (lowest at alpha 0.80) n/a.
- the grid is frozen from the development run; it was not widened on this data
Final grid: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

## 6. Sigma form used by the point policy (Phase 9 check, per model unit and segment)

| segment   |   B2 (constant CV) |   B2 (constant CV, check failed for both forms) |   B2-sqrt |   model versions |
|:----------|-------------------:|------------------------------------------------:|----------:|-----------------:|
| low       |                  0 |                                               1 |         4 |                5 |
| mid       |                  0 |                                               4 |         1 |                5 |
| high      |                  0 |                                               5 |         0 |                5 |

## 7. Total cost: minimum over each policy's own settings, per stockout-to-holding ratio rho (currency units, pooled; a sensitivity, not the headline)

| policy       | rho 4 (critical ratio 0.80)     | rho 9 (critical ratio 0.90)     | rho 19 (critical ratio 0.95)    | rho 99 (critical ratio 0.99)   |
|:-------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------|
| B3a          | 1,281 at 0.8 (edge of its grid) | 1,555 at 0.8 (edge of its grid) | 2,069 at 0.9                    | 3,819 at 0.95                  |
| B2 / B2-sqrt | 2,293 at 0.8 (edge of its grid) | 2,413 at 0.8 (edge of its grid) | 2,652 at 0.8 (edge of its grid) | 4,475 at 0.9                   |
| Quantile     | 1,213 at 0.8 (edge of its grid) | 1,503 at 0.8 (edge of its grid) | 1,931 at 0.9                    | 3,617 at 0.95                  |
| Naive        | 1,174 at 1.0 (edge of its grid) | 1,604 at 1.25                   | 2,146 at 1.5                    | 4,223 at 2.25                  |

## Addendum (written after the run, no re-run and no new test-window computation; arithmetic on the outputs above)

**Note on layout:** the heading "2. By velocity segment" appears twice above (a generator layout defect); the first is followed by the two-anchor reading paragraph, the second by the segment tables; no number is affected.

**How to read the comparison statistics.** Two statistics are reported, always labelled by their anchor, and the order is always B3a first (the fair benchmark), then B2 / B2-sqrt, then the naive rule:
- **Quantile-anchored matched inventory** (the pre-registered statistic): at each of the quantile policy's own alpha points, the comparator's inventory interpolated at that fill rate; undefined where the comparator's curve does not reach.
- **Comparator-anchored matched inventory** (added before the test-window run): at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate.
Neither is the mean of a common set of points, so their pooled means are not ordered against each other.

**Does the pooled interval for the B3a comparison exclude zero?** Yes for both statistics, but narrowly for one. Pooled, against B3a: comparator-anchored mean 16.0%, item-bootstrap interval [0.7%, 24.9%] (excludes zero by 0.7 percentage points; the mean is dominated by the alpha 0.99 setting, 46.3%, and the 0.80, 0.90 and 0.95 settings give 4.5%, 7.7% and 5.6%); quantile-anchored mean 12.7%, interval [9.5%, 24.4%] (two settings matched, 0.90 and 0.95). The intervals for the low- and high-velocity segments and for the peak include zero, and the settings 0.80 to 0.95 were not bootstrapped one by one, so **an advantage at moderate service levels is suggestive, not established**. Intervals reflect item sampling only, not variation between periods.

**The 0.99 end is steep for the quantile policy, and steeper for B3a.** Marginal average on-hand inventory (units per series-day) per percentage point of fill rate, from the curves in section 5:

| policy | 0.80 to 0.90 | 0.90 to 0.95 | 0.95 to 0.99 |
|---|---|---|---|
| Quantile policy | 1.10 | 2.17 | 10.92 |
| B3a (post-hoc) | 1.41 | 3.76 | 30.63 |
| B2 / B2-sqrt | 10.10 | 21.55 | 43.56 |

Going from 0.95 to 0.99 costs the quantile policy five times as much per fill-rate point as going from 0.90 to 0.95, so 0.99 is inefficient for it (prediction 2b, confirmed); B3a's step is eight times as steep and is about 2.8 times the quantile policy's, so much of the large reduction at 0.99 comes from B3a's tail being poorly priced, not only from the quantile policy being efficient.

**Cost sweep: results at a grid edge are unresolved.** The minima marked "edge of its grid" in section 7 are not minima of the underlying cost curve: for the naive rule at rho 4 (c = 1.00) and for the target-driven policies at alpha 0.80 (rho 4 and 9; B2 also at rho 19). The statement that the naive rule was cheapest at rho 4 (1,174 against the quantile policy's 1,213) is therefore **unresolved**; a post-hoc extension of the naive grid below c = 1.00 is logged in design.md. The target-driven policies' alpha grids cannot be extended (the quantile models exist only for 0.80, 0.90, 0.95, 0.99 in this range), so their rho 4 and 9 minima stay unresolved.
