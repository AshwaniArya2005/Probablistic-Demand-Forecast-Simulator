# Phase 10 sensitivity: random lead time Uniform{2, 3, 4}

Run 2026-09-21 at code commit `14a450d`. L = 3, R = 7 (P = 10); 10 run(s), 50 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
**Sensitivity: random lead time.** Each order's lead time is drawn Uniform{2, 3, 4} (mean 3); all policies plan with the nominal 3; 10 independent draws (seeds 0 to 9) are pooled; cycles keep their nominal windows. With R = 7 and leads of 2 to 4 days no order can overtake an earlier one, so no crossing occurs in this design (the engine supports it and is tested).

## 0. Expected outcomes (design.md, Phase 10 sensitivities entry), evaluated mechanically

- **HELD**: quantile policy: cycle service below the base case at every alpha
- **DID NOT HOLD**: quantile policy: fill rate below the base case at every alpha
- **HELD**: comparator-anchored mean reduction against B3a is positive

| policy       |   alpha |   fill base |   fill here |   cycle service base |   cycle service here |   avg on-hand base |   avg on-hand here |
|:-------------|--------:|------------:|------------:|---------------------:|---------------------:|-------------------:|-------------------:|
| B3a          |    0.8  |      0.9327 |      0.9306 |               0.8888 |               0.8712 |              10.89 |              10.89 |
| B3a          |    0.9  |      0.9636 |      0.9627 |               0.9488 |               0.9404 |              15.25 |              15.25 |
| B3a          |    0.95 |      0.9797 |      0.9794 |               0.9777 |               0.9741 |              21.3  |              21.29 |
| B3a          |    0.99 |      0.9955 |      0.9955 |               0.9972 |               0.9968 |              69.7  |              69.72 |
| B2 / B2-sqrt |    0.8  |      0.9764 |      0.976  |               0.9593 |               0.9535 |              24.46 |              24.46 |
| B2 / B2-sqrt |    0.9  |      0.986  |      0.9858 |               0.9796 |               0.9764 |              34.16 |              34.16 |
| B2 / B2-sqrt |    0.95 |      0.9898 |      0.9897 |               0.9875 |               0.9857 |              42.35 |              42.35 |
| B2 / B2-sqrt |    0.99 |      0.9934 |      0.9932 |               0.9933 |               0.9922 |              58.03 |              58.04 |
| Quantile     |    0.8  |      0.9318 |      0.9295 |               0.8848 |               0.8662 |              10.31 |              10.31 |
| Quantile     |    0.9  |      0.9611 |      0.9599 |               0.9405 |               0.9315 |              13.53 |              13.52 |
| Quantile     |    0.95 |      0.9768 |      0.9759 |               0.9707 |               0.9642 |              16.93 |              16.92 |
| Quantile     |    0.99 |      0.9958 |      0.9958 |               0.9971 |               0.9965 |              37.68 |              37.68 |

Comparator-anchored mean reduction of the quantile policy against B3a: base 16.0%, here 15.4%; against B2 / B2-sqrt: base 29.5%, here 29.0%.

**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.


## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 9.1%        | 15.5%        | n/a          | 2 of 4             | 12.3%               | [9.2%, 23.8%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | n/a          | n/a          | 0 of 4             | n/a                 | [27.6%, 40.9%]         | 50%                      |
| Naive (c x mu28 x P), widened grid        | 11.3%       | 17.0%       | 24.2%        | n/a          | 3 of 4             | 17.5%               | [14.3%, 31.3%]         | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.1%        | 7.4%        | 3.5%         | 46.4%        | 4 of 4             | 15.4%               | [0.5%, 23.7%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 30.3%       | 20.1%       | 26.1%        | 39.6%        | 4 of 4             | 29.0%               | [21.2%, 35.2%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 44.9%               | [12.7%, 51.6%]         | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 2.9%        | 11.0%       | -0.4%        | n/a          | 3 of 4             | 4.5%                | [-2.1%, 13.7%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 5.2%        | 8.7%        | 11.7%        | 5.7%         | 4 of 4             | 7.8%                | [-0.8%, 13.1%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 20 of 23           | 43.7%               | [11.3%, 50.2%]         | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 5.6%        | 9.8%        | 4.2%         | 47.4%        | 4 of 4             | 16.8%               | [-0.0%, 25.9%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 13.1%       | 12.2%       | 0.2%         | 11.3%        | 4 of 4             | 9.2%                | [1.2%, 15.2%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.0%               | [18.4%, 56.6%]         | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 4.8%        | 4.1%         | 48.7%        | 3 of 4             | 19.2%               | [-3.0%, 25.4%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 4.8%        | 18.2%       | 30.9%        | 47.1%        | 4 of 4             | 25.3%               | [7.7%, 35.4%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 17 of 23           | 36.5%               | [1.0%, 53.4%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.071 |       0.04  |        0.024 |        0.007 |
| all      | B2 / B2-sqrt |       0.153 |       0.076 |        0.036 |        0.002 |
| all      | Quantile     |       0.066 |       0.031 |        0.014 |        0.007 |
| peak     | B3a          |       0.079 |       0.039 |        0.023 |        0.006 |
| peak     | B2 / B2-sqrt |       0.145 |       0.073 |        0.032 |       -0.001 |
| peak     | Quantile     |       0.064 |       0.023 |        0.011 |        0.005 |
| rest     | B3a          |       0.069 |       0.041 |        0.024 |        0.007 |
| rest     | B2 / B2-sqrt |       0.156 |       0.077 |        0.037 |        0.003 |
| rest     | Quantile     |       0.067 |       0.034 |        0.015 |        0.007 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 0.0%        | 5.6%        | 2.9%         | n/a          | 3 of 4             | 2.8%                | [-8.4%, 19.6%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 22.2%       | 12.2%       | 17.3%        | 28.5%        | 4 of 4             | 20.1%               | [0.1%, 30.7%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 10 of 23           | 23.4%               | [-1.6%, 54.0%]         | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 8.5%        | 4.1%         | 48.2%        | 3 of 4             | 20.3%               | [9.7%, 25.7%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 31.6%       | 21.8%       | 28.0%        | 42.5%        | 4 of 4             | 31.0%               | [23.4%, 37.7%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 45.6%               | [39.8%, 53.1%]         | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9306 |          0.8712 |                                10.89 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9627 |          0.9404 |                                15.25 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9794 |          0.9741 |                                21.29 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9955 |          0.9968 |                                69.72 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.976  |          0.9535 |                                24.46 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.9858 |          0.9764 |                                34.16 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9897 |          0.9857 |                                42.35 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9932 |          0.9922 |                                58.04 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9295 |          0.8662 |                                10.31 |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9599 |          0.9315 |                                13.52 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9759 |          0.9642 |                                16.92 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.9958 |          0.9965 |                                37.68 |
| Naive (c x mu28 x P)                                              | c 1.00     |      0.8539 |          0.7162 |                                 7.19 |
| Naive (c x mu28 x P)                                              | c 1.25     |      0.9172 |          0.835  |                                10.28 |
| Naive (c x mu28 x P)                                              | c 1.50     |      0.9482 |          0.8994 |                                13.69 |
| Naive (c x mu28 x P)                                              | c 1.75     |      0.9643 |          0.9331 |                                17.28 |
| Naive (c x mu28 x P)                                              | c 2.00     |      0.9738 |          0.9548 |                                21.05 |
| Naive (c x mu28 x P)                                              | c 2.25     |      0.9802 |          0.9685 |                                24.98 |
| Naive (c x mu28 x P)                                              | c 2.50     |      0.9842 |          0.9752 |                                28.89 |
| Naive (c x mu28 x P)                                              | c 2.75     |      0.9865 |          0.9794 |                                32.8  |
| Naive (c x mu28 x P)                                              | c 3.00     |      0.9885 |          0.9844 |                                36.85 |
| Naive (c x mu28 x P)                                              | c 3.50     |      0.9905 |          0.989  |                                44.89 |
| Naive (c x mu28 x P)                                              | c 4.00     |      0.9916 |          0.9918 |                                53.17 |
| Naive (c x mu28 x P)                                              | c 4.50     |      0.9924 |          0.993  |                                61.52 |
| Naive (c x mu28 x P)                                              | c 5.00     |      0.993  |          0.9932 |                                69.96 |
| Naive (c x mu28 x P)                                              | c 5.50     |      0.9935 |          0.9933 |                                78.48 |
| Naive (c x mu28 x P)                                              | c 6.00     |      0.9941 |          0.9934 |                                87.06 |
| Naive (c x mu28 x P)                                              | c 6.50     |      0.9944 |          0.9938 |                                95.81 |
| Naive (c x mu28 x P)                                              | c 7.00     |      0.9945 |          0.9939 |                               104.37 |
| Naive (c x mu28 x P)                                              | c 7.50     |      0.9947 |          0.9941 |                               113.39 |
| Naive (c x mu28 x P)                                              | c 8.00     |      0.9948 |          0.9942 |                               122.23 |
| Naive (c x mu28 x P)                                              | c 8.50     |      0.995  |          0.9946 |                               131.21 |
| Naive (c x mu28 x P)                                              | c 9.00     |      0.995  |          0.9947 |                               140.11 |
| Naive (c x mu28 x P)                                              | c 9.50     |      0.9951 |          0.9947 |                               149.17 |
| Naive (c x mu28 x P)                                              | c 10.00    |      0.9951 |          0.9949 |                               158.18 |

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

| policy       | rho 4 (critical ratio 0.80)      | rho 9 (critical ratio 0.90)      | rho 19 (critical ratio 0.95)     | rho 99 (critical ratio 0.99)   |
|:-------------|:---------------------------------|:---------------------------------|:---------------------------------|:-------------------------------|
| B3a          | 12,898 at 0.8 (edge of its grid) | 15,755 at 0.8 (edge of its grid) | 20,907 at 0.9                    | 38,578 at 0.95                 |
| B2 / B2-sqrt | 22,955 at 0.8 (edge of its grid) | 24,180 at 0.8 (edge of its grid) | 26,631 at 0.8 (edge of its grid) | 45,175 at 0.9                  |
| Quantile     | 12,235 at 0.8 (edge of its grid) | 15,280 at 0.8 (edge of its grid) | 19,629 at 0.9                    | 37,429 at 0.95                 |
| Naive        | 11,907 at 1.0 (edge of its grid) | 16,236 at 1.25                   | 21,738 at 1.5                    | 42,333 at 2.25                 |
