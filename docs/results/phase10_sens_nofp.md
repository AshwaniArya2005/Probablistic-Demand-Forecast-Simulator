# Phase 10 sensitivity: no future-price inputs

Run 2026-09-21 at code commit `14a450d`. L = 3, R = 7 (P = 10); 1 run(s), 5 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
**Sensitivity: no future-price inputs.** Quantile and mean models refit without the planned-window price features (the Phase 7 `nofutprice` arm) by `ml/sim_rows.py testnofp`; B2 and B3a use the same refit point forecast.

## 0. Expected outcomes (design.md, Phase 10 sensitivities entry), evaluated mechanically

- **HELD**: quantile policy: cycle service within 0.01 of the base case at every alpha
- **HELD**: quantile policy: average on-hand within 5% of the base case at every alpha
- **DID NOT HOLD**: comparator-anchored mean reduction against B3a within 5 percentage points of the base case

| policy       |   alpha |   fill base |   fill here |   cycle service base |   cycle service here |   avg on-hand base |   avg on-hand here |
|:-------------|--------:|------------:|------------:|---------------------:|---------------------:|-------------------:|-------------------:|
| B3a          |    0.8  |      0.9327 |      0.9322 |               0.8888 |               0.8889 |              10.89 |              10.93 |
| B3a          |    0.9  |      0.9636 |      0.9638 |               0.9488 |               0.9493 |              15.25 |              15.35 |
| B3a          |    0.95 |      0.9797 |      0.9791 |               0.9777 |               0.9773 |              21.3  |              21.26 |
| B3a          |    0.99 |      0.9955 |      0.9956 |               0.9972 |               0.9971 |              69.7  |              74.76 |
| B2 / B2-sqrt |    0.8  |      0.9764 |      0.9763 |               0.9593 |               0.9591 |              24.46 |              24.68 |
| B2 / B2-sqrt |    0.9  |      0.986  |      0.9859 |               0.9796 |               0.9789 |              34.16 |              34.43 |
| B2 / B2-sqrt |    0.95 |      0.9898 |      0.9897 |               0.9875 |               0.9869 |              42.35 |              42.67 |
| B2 / B2-sqrt |    0.99 |      0.9934 |      0.9933 |               0.9933 |               0.9932 |              58.03 |              58.43 |
| Quantile     |    0.8  |      0.9318 |      0.9327 |               0.8848 |               0.8844 |              10.31 |              10.38 |
| Quantile     |    0.9  |      0.9611 |      0.9615 |               0.9405 |               0.9411 |              13.53 |              13.52 |
| Quantile     |    0.95 |      0.9768 |      0.9766 |               0.9707 |               0.9704 |              16.93 |              16.89 |
| Quantile     |    0.99 |      0.9958 |      0.996  |               0.9971 |               0.9975 |              37.68 |              37.71 |

Comparator-anchored mean reduction of the quantile policy against B3a: base 16.0%, here 22.1%; against B2 / B2-sqrt: base 29.5%, here 30.4%.

**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.


## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 5.7%        | 10.0%       | 16.7%        | n/a          | 3 of 4             | 10.8%               | [9.5%, 25.2%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | 32.4%        | n/a          | 1 of 4             | 32.4%               | [29.0%, 42.7%]         | 57%                      |
| Naive (c x mu28 x P), widened grid        | 11.9%       | 17.6%       | 25.1%        | n/a          | 3 of 4             | 18.2%               | [14.6%, 32.0%]         | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 8.6%        | 7.6%         | 50.1%        | 3 of 4             | 22.1%               | [1.9%, 27.1%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 31.8%       | 22.0%       | 27.5%        | 40.3%        | 4 of 4             | 30.4%               | [22.9%, 36.6%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 45.5%               | [13.2%, 52.6%]         | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.1%        | 12.0%       | 3.2%         | n/a          | 3 of 4             | 6.4%                | [0.1%, 18.1%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 6.3%        | 10.2%       | 12.6%        | 8.0%         | 4 of 4             | 9.3%                | [0.9%, 14.7%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 20 of 23           | 44.0%               | [13.0%, 51.5%]         | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 12.8%       | 11.3%        | 48.6%        | 3 of 4             | 24.2%               | [2.7%, 31.0%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 16.9%       | 19.4%       | 5.4%         | 12.6%        | 4 of 4             | 13.6%               | [4.1%, 21.0%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.8%               | [18.4%, 58.4%]         | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 5.4%        | 6.8%         | 53.3%        | 3 of 4             | 21.8%               | [-2.6%, 27.5%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 7.3%        | 19.7%       | 32.0%        | 47.7%        | 4 of 4             | 26.7%               | [9.6%, 36.9%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 19 of 23           | 40.9%               | [0.8%, 54.5%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.089 |       0.049 |        0.027 |        0.007 |
| all      | B2 / B2-sqrt |       0.159 |       0.079 |        0.037 |        0.003 |
| all      | Quantile     |       0.084 |       0.041 |        0.02  |        0.007 |
| peak     | B3a          |       0.092 |       0.048 |        0.026 |        0.006 |
| peak     | B2 / B2-sqrt |       0.151 |       0.075 |        0.032 |       -0.001 |
| peak     | Quantile     |       0.081 |       0.032 |        0.016 |        0.005 |
| rest     | B3a          |       0.088 |       0.05  |        0.028 |        0.008 |
| rest     | B2 / B2-sqrt |       0.162 |       0.08  |        0.038 |        0.004 |
| rest     | Quantile     |       0.086 |       0.044 |        0.022 |        0.008 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 0.3%        | 6.6%        | 8.4%         | n/a          | 3 of 4             | 5.1%                | [-6.2%, 19.8%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 23.4%       | 15.4%       | 18.3%        | 28.2%        | 4 of 4             | 21.3%               | [2.8%, 32.4%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 10 of 23           | 25.1%               | [0.3%, 54.2%]          | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 9.5%        | 7.4%         | 52.9%        | 3 of 4             | 23.3%               | [15.3%, 29.0%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 33.2%       | 23.3%       | 29.6%        | 43.4%        | 4 of 4             | 32.4%               | [24.6%, 39.4%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 46.2%               | [39.9%, 54.4%]         | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9322 |          0.8889 |                                10.93 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9638 |          0.9493 |                                15.35 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9791 |          0.9773 |                                21.26 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9956 |          0.9971 |                                74.76 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.9763 |          0.9591 |                                24.68 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.9859 |          0.9789 |                                34.43 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9897 |          0.9869 |                                42.67 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9933 |          0.9932 |                                58.43 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9327 |          0.8844 |                                10.38 |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9615 |          0.9411 |                                13.52 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9766 |          0.9704 |                                16.89 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.996  |          0.9975 |                                37.71 |
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
| B3a          | 1,289 at 0.8 (edge of its grid) | 1,566 at 0.8 (edge of its grid) | 2,081 at 0.9                    | 3,866 at 0.95                  |
| B2 / B2-sqrt | 2,316 at 0.8 (edge of its grid) | 2,436 at 0.8 (edge of its grid) | 2,677 at 0.8 (edge of its grid) | 4,526 at 0.9                   |
| Quantile     | 1,218 at 0.8 (edge of its grid) | 1,507 at 0.8 (edge of its grid) | 1,918 at 0.9                    | 3,609 at 0.95                  |
| Naive        | 1,174 at 1.0 (edge of its grid) | 1,604 at 1.25                   | 2,146 at 1.5                    | 4,223 at 2.25                  |
