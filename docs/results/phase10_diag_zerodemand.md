# Phase 10 diagnostic: excluding zero-demand cycles (hindsight)

Run 2026-09-21 at code commit `14a450d`. L = 3, R = 7 (P = 10); 1 run(s), 5 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
**Hindsight diagnostic, never the headline.** Every (cycle, series) cell whose realised demand over the protection interval t + 1 .. t + P was zero is dropped from every policy's metrics alike.

## 0. Expected outcomes (design.md, Phase 10 sensitivities entry), evaluated mechanically

- **HELD**: fill rate identical to the base case for every policy setting (dropped cells had no demand)
- **HELD**: cycle service below the base case for every quantile, B3a and B2 setting
- **HELD**: comparator-anchored mean reduction against B3a is positive

| policy       |   alpha |   fill base |   fill here |   cycle service base |   cycle service here |   avg on-hand base |   avg on-hand here |
|:-------------|--------:|------------:|------------:|---------------------:|---------------------:|-------------------:|-------------------:|
| B3a          |    0.8  |      0.9327 |      0.9327 |               0.8888 |               0.8719 |              10.89 |              11.14 |
| B3a          |    0.9  |      0.9636 |      0.9636 |               0.9488 |               0.941  |              15.25 |              15.8  |
| B3a          |    0.95 |      0.9797 |      0.9797 |               0.9777 |               0.9744 |              21.3  |              22.33 |
| B3a          |    0.99 |      0.9955 |      0.9955 |               0.9972 |               0.9968 |              69.7  |              74.88 |
| B2 / B2-sqrt |    0.8  |      0.9764 |      0.9764 |               0.9593 |               0.9532 |              24.46 |              26.06 |
| B2 / B2-sqrt |    0.9  |      0.986  |      0.986  |               0.9796 |               0.9765 |              34.16 |              36.62 |
| B2 / B2-sqrt |    0.95 |      0.9898 |      0.9898 |               0.9875 |               0.9856 |              42.35 |              45.55 |
| B2 / B2-sqrt |    0.99 |      0.9934 |      0.9934 |               0.9933 |               0.9923 |              58.03 |              62.64 |
| Quantile     |    0.8  |      0.9318 |      0.9318 |               0.8848 |               0.8673 |              10.31 |              10.52 |
| Quantile     |    0.9  |      0.9611 |      0.9611 |               0.9405 |               0.9315 |              13.53 |              13.76 |
| Quantile     |    0.95 |      0.9768 |      0.9768 |               0.9707 |               0.9662 |              16.93 |              17.11 |
| Quantile     |    0.99 |      0.9958 |      0.9958 |               0.9971 |               0.9966 |              37.68 |              39.24 |

Comparator-anchored mean reduction of the quantile policy against B3a: base 16.0%, here 17.6%; against B2 / B2-sqrt: base 29.5%, here 32.3%.

**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.


## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 10.7%       | 19.1%        | n/a          | 2 of 4             | 14.9%               | [11.1%, 26.3%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | 35.3%        | n/a          | 1 of 4             | 35.3%               | [31.7%, 45.1%]         | 58%                      |
| Naive (c x mu28 x P), widened grid        | 13.7%       | 20.1%       | 28.9%        | n/a          | 3 of 4             | 20.9%               | [16.7%, 34.5%]         | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.8%        | 9.5%        | 8.2%         | 47.9%        | 4 of 4             | 17.6%               | [2.2%, 27.0%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 34.6%       | 23.8%       | 29.0%        | 41.8%        | 4 of 4             | 32.3%               | [24.7%, 38.6%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 47.1%               | [15.5%, 54.3%]         | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 3.1%        | 13.6%       | 1.4%         | n/a          | 3 of 4             | 6.0%                | [-2.1%, 17.8%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 4.5%        | 8.6%        | 12.2%        | 6.4%         | 4 of 4             | 7.9%                | [-1.0%, 13.4%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 20 of 23           | 46.3%               | [14.7%, 53.4%]         | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 7.4%        | 12.5%       | 10.3%        | 49.2%        | 4 of 4             | 19.9%               | [3.1%, 30.2%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 15.8%       | 18.0%       | 5.1%         | 14.4%        | 4 of 4             | 13.3%               | [4.6%, 20.4%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 50.9%               | [21.9%, 60.9%]         | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 6.7%        | 9.6%         | 49.8%        | 3 of 4             | 22.0%               | [-1.3%, 27.9%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 9.5%        | 21.3%       | 32.9%        | 48.4%        | 4 of 4             | 28.0%               | [10.1%, 38.6%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 15 of 23           | 34.6%               | [3.1%, 55.7%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.072 |       0.041 |        0.024 |        0.007 |
| all      | B2 / B2-sqrt |       0.153 |       0.077 |        0.036 |        0.002 |
| all      | Quantile     |       0.067 |       0.032 |        0.016 |        0.007 |
| peak     | B3a          |       0.072 |       0.036 |        0.023 |        0.005 |
| peak     | B2 / B2-sqrt |       0.142 |       0.072 |        0.029 |       -0.003 |
| peak     | Quantile     |       0.059 |       0.019 |        0.01  |        0.004 |
| rest     | B3a          |       0.072 |       0.042 |        0.025 |        0.007 |
| rest     | B2 / B2-sqrt |       0.156 |       0.078 |        0.037 |        0.004 |
| rest     | Quantile     |       0.07  |       0.035 |        0.018 |        0.007 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 0.5%        | 9.1%        | 10.8%        | n/a          | 3 of 4             | 6.8%                | [-5.4%, 22.3%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 27.4%       | 18.5%       | 22.1%        | 30.9%        | 4 of 4             | 24.7%               | [5.3%, 35.7%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 10 of 23           | 28.9%               | [4.0%, 56.7%]          | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 10.0%       | 7.7%         | 49.8%        | 3 of 4             | 22.5%               | [10.6%, 28.3%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 35.9%       | 24.7%       | 30.5%        | 44.6%        | 4 of 4             | 33.9%               | [26.2%, 40.7%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 47.5%               | [41.3%, 55.7%]         | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9327 |          0.8719 |                                11.14 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9636 |          0.941  |                                15.8  |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9797 |          0.9744 |                                22.33 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9955 |          0.9968 |                                74.88 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.9764 |          0.9532 |                                26.06 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.986  |          0.9765 |                                36.62 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9898 |          0.9856 |                                45.55 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9934 |          0.9923 |                                62.64 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9318 |          0.8673 |                                10.52 |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9611 |          0.9315 |                                13.76 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9768 |          0.9662 |                                17.11 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.9958 |          0.9966 |                                39.24 |
| Naive (c x mu28 x P)                                              | c 1.00     |      0.8576 |          0.7059 |                                 7.36 |
| Naive (c x mu28 x P)                                              | c 1.25     |      0.9195 |          0.8326 |                                10.68 |
| Naive (c x mu28 x P)                                              | c 1.50     |      0.9497 |          0.8976 |                                14.37 |
| Naive (c x mu28 x P)                                              | c 1.75     |      0.9652 |          0.9323 |                                18.27 |
| Naive (c x mu28 x P)                                              | c 2.00     |      0.9742 |          0.9538 |                                22.34 |
| Naive (c x mu28 x P)                                              | c 2.25     |      0.9804 |          0.9678 |                                26.57 |
| Naive (c x mu28 x P)                                              | c 2.50     |      0.9842 |          0.9747 |                                30.78 |
| Naive (c x mu28 x P)                                              | c 2.75     |      0.9865 |          0.9785 |                                35    |
| Naive (c x mu28 x P)                                              | c 3.00     |      0.9885 |          0.9834 |                                39.35 |
| Naive (c x mu28 x P)                                              | c 3.50     |      0.9905 |          0.9883 |                                48.01 |
| Naive (c x mu28 x P)                                              | c 4.00     |      0.9916 |          0.9916 |                                56.93 |
| Naive (c x mu28 x P)                                              | c 4.50     |      0.9924 |          0.9929 |                                65.91 |
| Naive (c x mu28 x P)                                              | c 5.00     |      0.993  |          0.9929 |                                74.99 |
| Naive (c x mu28 x P)                                              | c 5.50     |      0.9936 |          0.9929 |                                84.17 |
| Naive (c x mu28 x P)                                              | c 6.00     |      0.9942 |          0.9932 |                                93.39 |
| Naive (c x mu28 x P)                                              | c 6.50     |      0.9945 |          0.9936 |                               102.81 |
| Naive (c x mu28 x P)                                              | c 7.00     |      0.9946 |          0.9936 |                               112.04 |
| Naive (c x mu28 x P)                                              | c 7.50     |      0.9948 |          0.9939 |                               121.72 |
| Naive (c x mu28 x P)                                              | c 8.00     |      0.9949 |          0.9942 |                               131.24 |
| Naive (c x mu28 x P)                                              | c 8.50     |      0.9951 |          0.9946 |                               140.89 |
| Naive (c x mu28 x P)                                              | c 9.00     |      0.9951 |          0.9946 |                               150.48 |
| Naive (c x mu28 x P)                                              | c 9.50     |      0.9952 |          0.9946 |                               160.24 |
| Naive (c x mu28 x P)                                              | c 10.00    |      0.9952 |          0.9946 |                               169.93 |

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
| B3a          | 1,165 at 0.8 (edge of its grid) | 1,439 at 0.8 (edge of its grid) | 1,920 at 0.9                    | 3,632 at 0.95                  |
| B2 / B2-sqrt | 2,121 at 0.8 (edge of its grid) | 2,240 at 0.8 (edge of its grid) | 2,479 at 0.8 (edge of its grid) | 4,254 at 0.9                   |
| Quantile     | 1,104 at 0.8 (edge of its grid) | 1,395 at 0.8 (edge of its grid) | 1,776 at 0.9                    | 3,419 at 0.95                  |
| Naive        | 1,099 at 1.0 (edge of its grid) | 1,510 at 1.25                   | 2,034 at 1.5                    | 4,045 at 2.25                  |
