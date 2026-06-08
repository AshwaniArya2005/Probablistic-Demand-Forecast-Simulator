# Phase 10 sensitivity: lead time 7 (protection interval 14)

Run 2026-09-21 at code commit `14a450d`. L = 7, R = 7 (P = 14); 1 run(s), 5 model unit(s); 100 items, 300 series.
Item cluster bootstrap: 10,000 resamples over items, seed 0; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
**Sensitivity: lead time 7** (P = 14, cycles t + 8 .. t + 14); model versions refit at horizon 14 by `ml/sim_rows.py testP14`; everything else as the primary run.

## 0. Expected outcomes (design.md, Phase 10 sensitivities entry), evaluated mechanically

- **HELD**: quantile policy: cycle service above target at every alpha
- **DID NOT HOLD**: comparator-anchored mean reduction against B3a is positive and within 8 percentage points of the base case
- **HELD**: B2 still over-protects: its cycle service exceeds target by more than 0.05 at alpha 0.80
- **HELD**: quantile policy holds more inventory than in the base case at every alpha

| policy       |   alpha |   fill base |   fill here |   cycle service base |   cycle service here |   avg on-hand base |   avg on-hand here |
|:-------------|--------:|------------:|------------:|---------------------:|---------------------:|-------------------:|-------------------:|
| B3a          |    0.8  |      0.9327 |      0.9234 |               0.8888 |               0.8891 |              10.89 |              13.89 |
| B3a          |    0.9  |      0.9636 |      0.9597 |               0.9488 |               0.9507 |              15.25 |              19.74 |
| B3a          |    0.95 |      0.9797 |      0.9781 |               0.9777 |               0.982  |              21.3  |              29.19 |
| B3a          |    0.99 |      0.9955 |      0.9944 |               0.9972 |               0.9972 |              69.7  |             130.13 |
| B2 / B2-sqrt |    0.8  |      0.9764 |      0.9761 |               0.9593 |               0.9636 |              24.46 |              37.31 |
| B2 / B2-sqrt |    0.9  |      0.986  |      0.9855 |               0.9796 |               0.9836 |              34.16 |              53.74 |
| B2 / B2-sqrt |    0.95 |      0.9898 |      0.9884 |               0.9875 |               0.9897 |              42.35 |              67.75 |
| B2 / B2-sqrt |    0.99 |      0.9934 |      0.9916 |               0.9933 |               0.9932 |              58.03 |              94.79 |
| Quantile     |    0.8  |      0.9318 |      0.9201 |               0.8848 |               0.8807 |              10.31 |              12.9  |
| Quantile     |    0.9  |      0.9611 |      0.9536 |               0.9405 |               0.9373 |              13.53 |              16.89 |
| Quantile     |    0.95 |      0.9768 |      0.9737 |               0.9707 |               0.968  |              16.93 |              21.36 |
| Quantile     |    0.99 |      0.9958 |      0.9944 |               0.9971 |               0.9973 |              37.68 |              50.16 |

Comparator-anchored mean reduction of the quantile policy against B3a: base 16.0%, here 5.8%; against B2 / B2-sqrt: base 29.5%, here 38.2%.

**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.


## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | n/a         | 9.9%        | 20.6%        | 61.4%        | 3 of 4             | 30.7%               | [12.5%, 31.1%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | n/a         | n/a         | n/a          | n/a          | 0 of 4             | n/a                 | [35.0%, 49.3%]         | 21%                      |
| Naive (c x mu28 x P), widened grid        | 9.5%        | 14.8%       | 26.1%        | n/a          | 3 of 4             | 16.8%               | [11.5%, 28.2%]         | 100%                     |

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.3%        | 7.6%        | 5.6%         | n/a          | 3 of 4             | 5.8%                | [0.6%, 28.3%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 33.7%       | 29.6%       | 38.3%        | 51.2%        | 4 of 4             | 38.2%               | [27.5%, 45.7%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.3%               | [12.2%, 55.8%]         | 100%                     |

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

**low**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 5.1%        | 8.6%        | -12.7%       | n/a          | 3 of 4             | 0.3%                | [-5.0%, 9.8%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 5.8%        | 10.0%       | 13.6%        | 6.7%         | 4 of 4             | 9.0%                | [0.3%, 14.3%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 41.9%               | [2.4%, 53.7%]          | 100%                     |

**mid**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 9.4%        | 11.0%       | -3.4%        | 54.0%        | 4 of 4             | 17.8%               | [-1.3%, 26.9%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 14.9%       | -1.5%       | 7.2%         | 25.2%        | 4 of 4             | 11.4%               | [1.3%, 21.5%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 50.8%               | [9.6%, 61.1%]          | 100%                     |

**high**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 1.8%        | 4.9%        | 14.9%        | 66.0%        | 4 of 4             | 21.9%               | [0.6%, 32.5%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 18.1%       | 31.7%       | 43.7%        | 57.7%        | 4 of 4             | 37.8%               | [15.7%, 46.6%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.0%               | [2.4%, 58.3%]          | 100%                     |

## 3. Achieved cycle service minus target alpha, by period (calibration check)

| period   | policy       |   alpha 0.8 |   alpha 0.9 |   alpha 0.95 |   alpha 0.99 |
|:---------|:-------------|------------:|------------:|-------------:|-------------:|
| all      | B3a          |       0.089 |       0.051 |        0.032 |        0.007 |
| all      | B2 / B2-sqrt |       0.164 |       0.084 |        0.04  |        0.003 |
| all      | Quantile     |       0.081 |       0.037 |        0.018 |        0.007 |
| peak     | B3a          |       0.103 |       0.053 |        0.028 |        0.006 |
| peak     | B2 / B2-sqrt |       0.156 |       0.076 |        0.033 |       -0.001 |
| peak     | Quantile     |       0.082 |       0.041 |        0.014 |        0.006 |
| rest     | B3a          |       0.086 |       0.05  |        0.033 |        0.008 |
| rest     | B2 / B2-sqrt |       0.165 |       0.086 |        0.042 |        0.004 |
| rest     | Quantile     |       0.08  |       0.036 |        0.019 |        0.008 |

## 4. By period, inventory reduction at matched fill rate (comparator-anchored)

**peak**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.3%        | 8.6%        | 8.1%         | n/a          | 3 of 4             | 7.0%                | [-8.7%, 27.6%]         | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 34.8%       | 26.0%       | 29.6%        | n/a          | 3 of 4             | 30.1%               | [6.0%, 53.4%]          | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 13 of 23           | 35.2%               | [-3.1%, 67.7%]         | 100%                     |

**rest**

| comparator                                | alpha 0.8   | alpha 0.9   | alpha 0.95   | alpha 0.99   | settings matched   | mean over matched   | 95% interval (items)   | resamples with a match   |
|:------------------------------------------|:------------|:------------|:-------------|:-------------|:-------------------|:--------------------|:-----------------------|:-------------------------|
| B3a (post-hoc, the fair benchmark)        | 4.4%        | 7.4%        | 4.9%         | 63.2%        | 4 of 4             | 20.0%               | [3.9%, 31.2%]          | 100%                     |
| B2 / B2-sqrt (point policy, normal sigma) | 32.3%       | 30.0%       | 40.0%        | 53.8%        | 4 of 4             | 39.0%               | [32.1%, 45.9%]         | 100%                     |
| Naive (c x mu28 x P), widened grid        | -           | -           | -            | -            | 21 of 23           | 48.2%               | [41.9%, 55.1%]         | 100%                     |

## 5. Policy curves (pooled over runs)

| policy                                                            | setting    |   fill rate |   cycle service |   avg on-hand (units per series-day) |
|:------------------------------------------------------------------|:-----------|------------:|----------------:|-------------------------------------:|
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.8  |      0.9234 |          0.8891 |                                13.89 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.9  |      0.9597 |          0.9507 |                                19.74 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.95 |      0.9781 |          0.982  |                                29.19 |
| B3a (point forecast + empirical residual quantiles, post-hoc)     | alpha 0.99 |      0.9944 |          0.9972 |                               130.13 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.8  |      0.9761 |          0.9636 |                                37.31 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.9  |      0.9855 |          0.9836 |                                53.74 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.95 |      0.9884 |          0.9897 |                                67.75 |
| B2 / B2-sqrt: point policy, normal sigma (form per segment below) | alpha 0.99 |      0.9916 |          0.9932 |                                94.79 |
| Quantile policy (headline)                                        | alpha 0.8  |      0.9201 |          0.8807 |                                12.9  |
| Quantile policy (headline)                                        | alpha 0.9  |      0.9536 |          0.9373 |                                16.89 |
| Quantile policy (headline)                                        | alpha 0.95 |      0.9737 |          0.968  |                                21.36 |
| Quantile policy (headline)                                        | alpha 0.99 |      0.9944 |          0.9973 |                                50.16 |
| Naive (c x mu28 x P)                                              | c 1.00     |      0.826  |          0.7193 |                                 8.17 |
| Naive (c x mu28 x P)                                              | c 1.25     |      0.9073 |          0.8496 |                                12.6  |
| Naive (c x mu28 x P)                                              | c 1.50     |      0.9457 |          0.912  |                                17.55 |
| Naive (c x mu28 x P)                                              | c 1.75     |      0.9641 |          0.9467 |                                22.83 |
| Naive (c x mu28 x P)                                              | c 2.00     |      0.9726 |          0.9604 |                                27.99 |
| Naive (c x mu28 x P)                                              | c 2.25     |      0.9798 |          0.9759 |                                33.93 |
| Naive (c x mu28 x P)                                              | c 2.50     |      0.9829 |          0.9816 |                                39.52 |
| Naive (c x mu28 x P)                                              | c 2.75     |      0.9851 |          0.9855 |                                45.33 |
| Naive (c x mu28 x P)                                              | c 3.00     |      0.9864 |          0.9877 |                                50.97 |
| Naive (c x mu28 x P)                                              | c 3.50     |      0.9881 |          0.9904 |                                62.73 |
| Naive (c x mu28 x P)                                              | c 4.00     |      0.9893 |          0.9919 |                                74.37 |
| Naive (c x mu28 x P)                                              | c 4.50     |      0.99   |          0.9929 |                                86.83 |
| Naive (c x mu28 x P)                                              | c 5.00     |      0.9902 |          0.9931 |                                98.95 |
| Naive (c x mu28 x P)                                              | c 5.50     |      0.9904 |          0.9933 |                               111.44 |
| Naive (c x mu28 x P)                                              | c 6.00     |      0.9906 |          0.9936 |                               123.64 |
| Naive (c x mu28 x P)                                              | c 6.50     |      0.9907 |          0.9936 |                               136.4  |
| Naive (c x mu28 x P)                                              | c 7.00     |      0.9908 |          0.9937 |                               148.91 |
| Naive (c x mu28 x P)                                              | c 7.50     |      0.9911 |          0.9937 |                               161.82 |
| Naive (c x mu28 x P)                                              | c 8.00     |      0.9912 |          0.9939 |                               174.28 |
| Naive (c x mu28 x P)                                              | c 8.50     |      0.9914 |          0.9943 |                               187.45 |
| Naive (c x mu28 x P)                                              | c 9.00     |      0.9915 |          0.9944 |                               200.39 |
| Naive (c x mu28 x P)                                              | c 9.50     |      0.9916 |          0.9944 |                               213.56 |
| Naive (c x mu28 x P)                                              | c 10.00    |      0.9917 |          0.9944 |                               226.44 |

Naive c-grid widening (rule 3): start [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]; top target (highest fill among the other policies at alpha 0.99) n/a; bottom target (lowest at alpha 0.80) n/a.
- the grid is frozen from the development run; it was not widened on this data
Final grid: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

## 6. Sigma form used by the point policy (Phase 9 check, per model unit and segment)

| segment   |   B2 (constant CV) |   B2 (constant CV, check failed for both forms) |   B2-sqrt |   model versions |
|:----------|-------------------:|------------------------------------------------:|----------:|-----------------:|
| low       |                  0 |                                               1 |         4 |                5 |
| mid       |                  0 |                                               5 |         0 |                5 |
| high      |                  0 |                                               5 |         0 |                5 |

## 7. Total cost: minimum over each policy's own settings, per stockout-to-holding ratio rho (currency units, pooled; a sensitivity, not the headline)

| policy       | rho 4 (critical ratio 0.80)     | rho 9 (critical ratio 0.90)     | rho 19 (critical ratio 0.95)    | rho 99 (critical ratio 0.99)    |
|:-------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|
| B3a          | 1,630 at 0.8 (edge of its grid) | 1,954 at 0.8 (edge of its grid) | 2,603 at 0.8 (edge of its grid) | 4,813 at 0.95                   |
| B2 / B2-sqrt | 3,420 at 0.8 (edge of its grid) | 3,543 at 0.8 (edge of its grid) | 3,789 at 0.8 (edge of its grid) | 5,758 at 0.8 (edge of its grid) |
| Quantile     | 1,526 at 0.8 (edge of its grid) | 1,879 at 0.8 (edge of its grid) | 2,415 at 0.9                    | 4,488 at 0.95                   |
| Naive        | 1,397 at 1.0 (edge of its grid) | 1,934 at 1.25                   | 2,609 at 1.5                    | 5,096 at 2.25                   |
