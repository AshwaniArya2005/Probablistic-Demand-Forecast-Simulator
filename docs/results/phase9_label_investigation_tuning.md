# Phase 9 follow-up: why realised shortfall among HIGH rows is below one half

Run 2026-09-21 at code commit `aaed13f`. Report-only, tuning folds only, touch log unchanged. Label definition unchanged (design.md, Phase 9 pre-registration).
For a calibrated integer model, IP < Q50 implies P(demand > IP) >= P(demand >= Q50) >= 0.5, so realised shortfall well below one half in HIGH rows means the
model's median is too high **on those rows**, or the rows are not what the label assumes. HIGH rows are pooled over the c = 1, 1.5, 2 reference positions
(P in {10, 14}); `y_at_or_above_Q50` is the share of those rows whose demand reached the ceiled median.

## Realised shortfall among HIGH rows by the median

|                                 |    n |   realised_shortfall |   y_at_or_above_Q50 |
|:--------------------------------|-----:|---------------------:|--------------------:|
| (1.0, '0 < q50 <= 1 (Q50 = 1)') | 2009 |                0.174 |               0.174 |
| (1.0, '1 < q50 <= 2 (Q50 = 2)') |  371 |                0.496 |               0.453 |
| (1.0, '2 < q50 <= 5')           | 1071 |                0.537 |               0.503 |
| (1.0, 'q50 > 5')                | 4096 |                0.558 |               0.494 |
| (1.5, '0 < q50 <= 1 (Q50 = 1)') | 2009 |                0.174 |               0.174 |
| (1.5, '1 < q50 <= 2 (Q50 = 2)') |  259 |                0.49  |               0.429 |
| (1.5, '2 < q50 <= 5')           |  304 |                0.586 |               0.503 |
| (1.5, 'q50 > 5')                |  377 |                0.679 |               0.515 |
| (2.0, '0 < q50 <= 1 (Q50 = 1)') | 2009 |                0.174 |               0.174 |
| (2.0, '1 < q50 <= 2 (Q50 = 2)') |  138 |                0.493 |               0.377 |
| (2.0, '2 < q50 <= 5')           |  174 |                0.598 |               0.5   |
| (2.0, 'q50 > 5')                |  194 |                0.675 |               0.505 |

## by the gap Q50 - IP (units the reference position sits below the ceiled median)

|                   |    n |   realised_shortfall |   y_at_or_above_Q50 |
|:------------------|-----:|---------------------:|--------------------:|
| (1.0, 'gap 1')    | 5064 |                0.368 |               0.368 |
| (1.0, 'gap 2-3')  | 1488 |                0.571 |               0.484 |
| (1.0, 'gap 4-10') |  768 |                0.65  |               0.514 |
| (1.0, 'gap > 10') |  227 |                0.806 |               0.458 |
| (1.5, 'gap 1')    | 2418 |                0.229 |               0.229 |
| (1.5, 'gap 2-3')  |  315 |                0.597 |               0.463 |
| (1.5, 'gap 4-10') |  143 |                0.748 |               0.531 |
| (1.5, 'gap > 10') |   73 |                0.849 |               0.438 |
| (2.0, 'gap 1')    | 2177 |                0.198 |               0.198 |
| (2.0, 'gap 2-3')  |  213 |                0.587 |               0.437 |
| (2.0, 'gap 4-10') |   91 |                0.769 |               0.549 |
| (2.0, 'gap > 10') |   34 |                0.824 |               0.412 |

## by outage status at the origin (zero run of 14 or more days in the last 91)

|                                    |    n |   realised_shortfall |   y_at_or_above_Q50 |
|:-----------------------------------|-----:|---------------------:|--------------------:|
| (1.0, 'in a zero run of 14+ days') | 4012 |                0.372 |               0.327 |
| (1.0, 'no long zero run')          | 3535 |                0.538 |               0.5   |
| (1.5, 'in a zero run of 14+ days') | 2873 |                0.297 |               0.262 |
| (1.5, 'no long zero run')          |   76 |                0.763 |               0.724 |
| (2.0, 'in a zero run of 14+ days') | 2506 |                0.257 |               0.231 |
| (2.0, 'no long zero run')          |    9 |                1     |               0.778 |

## by velocity segment

|               |    n |   realised_shortfall |   y_at_or_above_Q50 |
|:--------------|-----:|---------------------:|--------------------:|
| (1.0, 'high') | 3062 |                0.498 |               0.437 |
| (1.0, 'low')  | 2218 |                0.353 |               0.331 |
| (1.0, 'mid')  | 2267 |                0.48  |               0.445 |
| (1.5, 'high') |  795 |                0.347 |               0.284 |
| (1.5, 'low')  | 1426 |                0.287 |               0.267 |
| (1.5, 'mid')  |  728 |                0.31  |               0.276 |
| (2.0, 'high') |  692 |                0.298 |               0.25  |
| (2.0, 'low')  | 1194 |                0.25  |               0.234 |
| (2.0, 'mid')  |  629 |                0.237 |               0.215 |

## The median itself, all origins with q50 > 0: share with demand at or above Q50 (should be about 0.5 or more)

| segment   |    n |   y_at_or_above_Q50 |
|:----------|-----:|--------------------:|
| high      | 9423 |               0.491 |
| low       | 9283 |               0.454 |
| mid       | 9497 |               0.489 |

## Share of origins with zero demand, by the size of q50 (all rows, P in {10, 14})

| q50     |     n |   share_y_zero |
|:--------|------:|---------------:|
| q50 = 0 |   597 |          0.921 |
| 0-0.5   |  1982 |          0.835 |
| 0.5-1   |  1195 |          0.486 |
| 1-2     |  2084 |          0.277 |
| 2-5     |  5474 |          0.115 |
| >5      | 17468 |          0.02  |
