"""The pre-registered Phase 6 decision rules (docs/design.md section 12) as pure functions of mean WAPE values."""
TIE = 0.002            # configs within this WAPE of the best: take the simplest
YEAR_AGO_GAIN = 0.003  # sum_364_P is kept only if it lowers XGBoost's mean WAPE by at least this much
ID_GAIN = 0.003        # a larger id set is used only if it beats the smaller one by at least this much
ID_ARMS = ("no_ids", "no_item", "all")      # simplest first


def simplest_within(items, wapes, tol=TIE):
    """items listed simplest first; returns the first whose mean WAPE is within `tol` of the lowest"""
    best = min(wapes)
    return next(i for i, w in zip(items, wapes) if w <= best + tol)


def choose_floor(wape_by_floor, default=1 / 28):
    """lowest mean WAPE; exact ties (to 4 decimals) go to the default floor"""
    best = min(wape_by_floor.values())
    ties = [f for f, w in wape_by_floor.items() if round(w, 4) == round(best, 4)]
    return default if default in ties else min(ties)


def keep_year_ago(wape_with, wape_without, gain=YEAR_AGO_GAIN):
    return wape_with <= wape_without - gain


def choose_id_arm(wape_by_arm, gain=ID_GAIN):
    """start from no ids; move to a larger id set only if it is better by at least `gain` than the arm chosen so far"""
    chosen = ID_ARMS[0]
    for nxt in ID_ARMS[1:]:
        if wape_by_arm[chosen] - wape_by_arm[nxt] >= gain:
            chosen = nxt
    return chosen


# ---- Phase 7 (relative tolerances on scaled pinball) ----
REL_TIE = 0.005        # configs / floors within 0.5% relative of the best are tied: take the simpler / the default
REL_ADOPT = 0.01       # normalisation must win by at least 1% relative on BOTH the mean and the median over series
REL_ADD = 0.005        # a feature group is added only if it lowers mean scaled pinball by at least 0.5% relative


def simplest_within_rel(items, values, rel=REL_TIE):
    best = min(values)
    return next(i for i, v in zip(items, values) if v <= best * (1 + rel))


def choose_floor_rel(value_by_floor, default=1 / 14, rel=REL_TIE):
    """floors within `rel` of the best are tied and the default wins if it is among them; otherwise the best floor"""
    best = min(value_by_floor.values())
    tied = [f for f, v in value_by_floor.items() if v <= best * (1 + rel)]
    return default if default in tied else min(value_by_floor, key=value_by_floor.get)


def adopt_normalised(raw_mean, raw_median, norm_mean, norm_median, rel=REL_ADOPT):
    """both the mean and the median over series must improve by at least `rel`: one tiny-scale series must not decide it"""
    return norm_mean <= raw_mean * (1 - rel) and norm_median <= raw_median * (1 - rel)


def add_group(base, with_group, rel=REL_ADD):
    return with_group <= base * (1 - rel)
