# Features (Phase 4)

Built by `ml/features.py`, checked by `ml/test_features.py`. One row per series x origin day `t`, all features known at the end of day `t`. `columns(P)` returns the 39 model inputs for horizon `P` in {7, 10, 14}; `columns(P, future_price=False)` drops the two future-price columns (the ablation in design section 4).

**Rows.** An origin exists when the series has been on sale for at least 91 days (`age_days >= 90`) and the 7-day target is complete (`t <= 2016-05-15`). 480,771 rows; the 25 Sunday review origins of the test window (22 Nov 2015 to 8 May 2016) are 7,500 of them. Training may use every day; evaluation uses the Sunday origins.

**Target** `y_p{P}` = raw sales on days `t+1..t+P` (closed-day sales, 0-7 units, are included because they are the actual sales). It is the only column built from future sales and is never in `columns(P)`.

**Conventions.** "Open days" excludes closed days (the Dec 25 calendar rule), whose near-zero sales are not demand. Rolling statistics over `w` days need `w-1` observations. Before a series' first `sell_price` week there is no data (NaN), and origins start 91 days after it. Trees handle the remaining NaNs (first-year `sum_364`).

## History features (data on or before `t`)

| Feature | Definition | Leakage note |
|---|---|---|
| `lag_0..lag_6` | sales on day `t-k` | Day `t` is complete at the end of day `t`. Only past or current days. |
| `mean_w`, `std_w` (w = 7, 14, 28, 91) | mean and sample std of open-day sales over `t-w+1..t` | Window ends at `t`. |
| `zero_frac_28`, `zero_frac_91` | share of open days with zero sales over the window | Window ends at `t`. |
| `days_since_sale` | days from the last sale on or before `t` to `t` (0 = sold today); if never sold, the series age | Looks back only. |
| `zero_run_91` | longest run of consecutive zero-sale open days inside `t-90..t`, capped at 56; closed days neither extend nor break a run | Window ends at `t`. Cap 56 is a choice, changeable on the tuning folds only. Design section 12: the answer to unfiltered stock-out runs. |
| `age_days` | `t` minus the first `sell_price` day | The first price day is on or before `t` for every origin (origins need 91+ days), so it is known. It is the same first-price rule that drops pre-launch rows. |
| `sum_last_{P}` | sales over `t-P+1..t` | Ends at `t`. |
| `sum_364_{P}` | sales over `t-363..t-364+P`, the same weekdays one year earlier | Ends at `t-364+P <= t` since `P <= 14`. |
| `price` | `sell_price` on day `t` | The current price is known at order time. |
| `price_rel_now` | `price / median(prices over the last 182 days ending at t)` (needs 91 days) | Median uses prices `<= t` only. Window length is a Phase 4 choice, tuning folds only. |
| `origin_dow` | weekday of `t` | Calendar. Constant (Sunday) at evaluation, varies in training rows. |
| `item_id`, `store_id`, `dept_id`, `cat_id` | identifiers as categories | Static labels, no target information. `item_id` lets a model memorise an item, which is intended (a global model with item effects), and is why evaluation is on later dates, not held-out items. |

## Future-window covariates (window `t+1..t+P`, one column set per P, suffix `_p{P}`)

These use dates after `t` on purpose (design section 4, known-prices assumption). They are functions of the calendar or the price plan and never of sales.

| Feature | Definition | Leakage note |
|---|---|---|
| `n_snap_p{P}` | number of SNAP days (CA schedule) in the window | The SNAP calendar is published in advance. |
| `n_event_p{P}`, `n_sporting_p{P}`, `n_cultural_p{P}`, `n_national_p{P}`, `n_religious_p{P}` | number of event days, and by event type (either event slot) | The event calendar is known in advance. |
| `n_weekend_p{P}` | number of Saturdays and Sundays | Pure calendar. |
| `n_closed_p{P}` | number of closed days (Dec 25 rule) | The rule was derived from 2011-2014 sales (all before the tuning cutoff) and is a function of the date only, so no sales inside the forecast window feed it. The test asserts the flag is exactly the Dec 25 rule. |
| `price_mean_rel_p{P}`, `price_min_rel_p{P}` | mean and minimum of the `sell_price` over the window, divided by the trailing median at `t` | **The one real assumption:** future prices are treated as the plan. Realised prices stand in for planned ones, which flatters ML relative to the naive rule. The denominator uses prices `<= t` only. Dropped in the no-future-price ablation. |

## Deliberately not features

- **Velocity segment.** Its label uses sales up to 2015-08-30, which is the future for any origin before that. It is for sampling, pooling and reporting only. The test asserts it is absent.
- **Anything computed from sales after `t`** other than the target. The test checks this by scrambling sales after an origin and rebuilding: no feature at or before that origin moves, while the targets whose windows reach past it do move (so the test can see a leak).
- **Cross-store zero-run and named-event features** are built but opt-in: they enter a model only if their Phase 7 tuning-fold ablation passes (see the last section).

## Checks (`uv run pytest tests/test_features.py`; `-m realdata` repeats them on the real panel)

1. Structure: no target column in model inputs, the ablation drops exactly the two future-price columns, no segment feature, the closed flag is only Dec 25.
2. Brute-force reference: 85 (series, origin) pairs, including origins around Dec 25 and the 1,578-day-lead series, recomputed with plain loops from the panel for all three horizons; 9,180 feature and target values match.
3. Perturbation (origin 2014-06-15): scrambling sales after it changes no feature at or before it; scrambling prices after it changes no feature except the future-price windows that reach past it (and prices never change the target).

## Phase 7 candidate groups (opt-in; not part of `columns(P)`)

Pre-registered in design.md section 12 (amendment to the Phase 7 pre-registration). They are always built into the features table but only enter a model through `columns(P, ev=..., xs=True)`, and only if their tuning-fold ablation passes (added iff mean scaled pinball falls by at least 0.5% relative). Both have brute-force reference and sales-perturbation leakage tests in `tests/test_features.py`.

| Feature | Definition | Leakage note |
|---|---|---|
| `ev_<name>_p{P}` (30 named events x 3 horizons) | 1 if a named calendar event (`event_name_1` or `_2`) falls in `t+1..t+P`, else 0. Events: those occurring on at least 3 dates on or before the tuning cutoff (all 30: Easter, Christmas, Thanksgiving, SuperBowl, ...) | The event calendar is public in advance. The list depends on calendar dates only, never on sales. Unchanged by scrambling sales or prices (tested). |
| `other_zero_run_91` | mean over the item's *other* stores on sale at `t` of their `zero_run_91` (capped 56) | Uses sibling stores' history on or before `t` only. NaN if no other store is on sale (an item with a single store). Unchanged when sales after `t` are scrambled (tested). |
| `other_zero_28` | number of the item's other stores on sale at `t` (0 to 2) whose sales over the last 28 open days sum to zero (needs 27 observations) | Same: sibling history on or before `t`. NaN if no sibling has a defined value. |
