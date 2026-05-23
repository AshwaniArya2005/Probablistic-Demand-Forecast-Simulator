"""Phase 8 explanations (design.md section 12: Phase 8 pre-registration and its correction). Exact per-quantile TreeSHAP contributions of the quantile
model (`XGBQ.contributions`), converted to units, grouped into themes, turned into sentences. Contributions are model associations, not causal effects.
Guards: only Sunday review origins are explained; origins whose target window passes the tuning cutoff are refused unless the caller passes
`allow_test_window=True` together with a reference to the touch-log entry that records the final frozen run."""
import re

import numpy as np
import pandas as pd

from folds import assert_tuning_only

# theme -> subject phrase used in sentences ("the price (price is 12% below usual)" is built separately)
THEMES = {"recent sales level": "the recent sales level", "zero-sales stretches": "the days-without-sales pattern",
          "same period last year": "the same period last year", "price vs usual": "the price", "price level": "the item's price level",
          "calendar": "the calendar", "events": "the event calendar", "other factors": "the remaining-factors group", "series identity": "the item-store identity"}
# planner sentences never interpret `age_days`: it sits in the neutral theme "other factors"; only global tables and charts name it, and say it is not interpreted
GLOBAL_LABEL = {"other factors": "series age (not interpreted)"}
BANNED = re.compile(r"\b(promotion|promo|promotions|discount|on sale|because|caused|due to)\b", re.I)
VERB_UP, VERB_DOWN = "raises", "lowers"
MIN_ABS_UNITS, TOP = 0.05, 3


def theme_of(name):
    """every model input belongs to exactly one theme; an unknown name is an error, never a silent 'other'"""
    if re.fullmatch(r"lag_\d+|(mean|std)_\d+|sum_last_\d+", name):
        return "recent sales level"
    if re.fullmatch(r"zero_frac_\d+|days_since_sale|zero_run_91|other_zero_run_91|other_zero_28", name):
        return "zero-sales stretches"
    if re.fullmatch(r"sum_364_\d+", name):
        return "same period last year"
    if name == "price":
        return "price level"                                  # the absolute price is an item-type proxy, not a deviation from usual
    if re.fullmatch(r"price_rel_now|price_(mean|min)_rel_p\d+", name):
        return "price vs usual"
    if re.fullmatch(r"n_(snap|weekend|closed)_p\d+|origin_dow", name):
        return "calendar"
    if re.fullmatch(r"n_(event|sporting|cultural|national|religious)_p\d+|ev_.+_p\d+", name):
        return "events"
    if name == "age_days":
        return "other factors"
    if name in ("item_id", "store_id", "dept_id", "cat_id"):
        return "series identity"
    raise KeyError(f"feature '{name}' is not assigned to a theme")


def ratio_to_units(phi_ratio, scale):
    """contributions in the normalised (ratio) space times the row scale = contributions in units"""
    return np.asarray(phi_ratio, float) * np.asarray(scale, float)[..., None]


def theme_contributions(phi, names):
    """(n, k, F) feature contributions -> (n, k, T) theme contributions (sums, so they stay additive), theme names in THEMES order"""
    order = list(THEMES)
    M = np.zeros((len(names), len(order)))
    for j, nm in enumerate(names):
        M[j, order.index(theme_of(nm))] = 1.0
    return np.asarray(phi, float) @ M, order


def price_words(rel_now, rel_window, rel_min=None):
    """'price is X% below usual' / 'above usual' / 'about usual', from whichever of the current price, the planned window mean and the planned window
    minimum (each relative to the usual price) deviates most"""
    rel = max([rel_now, rel_window] + ([] if rel_min is None else [rel_min]), key=lambda r: abs(1 - r))
    pct = round(abs(1 - rel) * 100)
    if pct < 1:
        return "price is about usual"
    return f"price is {pct}% {'below' if rel < 1 else 'above'} usual"


def clauses(theme_units, themes, price_text=None, top=TOP, min_abs=MIN_ABS_UNITS):
    """the top themes by absolute contribution: [(subject, verb, amount)]; the verb is decided by the sign and nothing else"""
    idx = [i for i in np.argsort(-np.abs(theme_units), kind="stable") if abs(theme_units[i]) >= min_abs][:top]
    out = []
    for i in idx:
        subject = THEMES[themes[i]]
        if themes[i] == "price vs usual" and price_text:
            subject = f"the price ({price_text})"
        out.append((subject, VERB_UP if theme_units[i] > 0 else VERB_DOWN, abs(float(theme_units[i]))))
    return out


def sentence(label, value, baseline, P, cl):
    head = f"{label}: {max(value, 0.0):.1f} units over the next {P} days (typical {baseline:.1f})"
    if value < 0:
        head += ", floored at 0"
    body = "; ".join(f"{s} {v} it by {a:.1f}" for s, v, a in cl)
    text = head + (". " + body[0].upper() + body[1:] + "." if body else ".")
    assert not BANNED.search(text), text
    return text


def explain_rows(model, rows, P, alpha, top=TOP, allow_test_window=False, touch_log_ref=None):
    """explain the P50 and the `alpha` service-level quantile for each row (Sunday review origins only). Returns a list of dicts per row:
    {row, p50: {text, value, baseline, themes}, service: {...}}. `rows` needs a `date` column and the model's inputs (and price_rel_now,
    price_mean_rel_p{P}, price_min_rel_p{P} for the price wording)."""
    d = pd.DatetimeIndex(rows["date"])
    if (d.dayofweek != 6).any():
        raise ValueError("only Sunday review origins are explained")
    if allow_test_window:
        if not touch_log_ref:
            raise ValueError("test-window explanations need a reference to their touch-log entry (design.md section 14)")
    else:
        assert_tuning_only(d, P)
    c = model.contributions(rows, [0.50, alpha])
    theme_units, themes = theme_contributions(c["phi"], c["names"])
    out = []
    for i in range(len(rows)):
        r = rows.iloc[i]
        ptxt = (price_words(float(r["price_rel_now"]), float(r[f"price_mean_rel_p{P}"]), float(r[f"price_min_rel_p{P}"]))
                if {"price_rel_now", f"price_mean_rel_p{P}", f"price_min_rel_p{P}"} <= set(rows.columns) else None)
        rec = {"row": i}
        for k, (key, label) in enumerate((("p50", "P50 forecast"), ("service", f"P{int(round(alpha * 100))} order-up-to quantile"))):
            cl = clauses(theme_units[i, k], themes, ptxt, top)
            rec[key] = dict(text=sentence(label, float(c["value"][i, k]), float(c["bias"][i, k]), P, cl), value=float(c["value"][i, k]),
                            baseline=float(c["bias"][i, k]), themes=dict(zip(themes, map(float, theme_units[i, k]))), source=int(c["source"][i, k]))
        out.append(rec)
    return out
