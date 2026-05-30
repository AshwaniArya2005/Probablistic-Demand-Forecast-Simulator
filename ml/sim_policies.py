"""Phase 10: order-up-to levels S for the four policies of docs/design.md section 12 (Phase 10 pre-run rules, item 4) from a forecast table.
tab: one row per (id, review date) with columns id, date, segment, mu28, scale, yhat, q80, q90, q95, q99 for one horizon P. cal: the calibration-window rows of the
same model version (y, yhat, scale, segment). Nothing here reads sales or simulated stock."""
import numpy as np
import pandas as pd

import conformal
import policy
import versions

SEGS = ("low", "mid", "high")
QCOL = {0.80: "q80", 0.90: "q90", 0.95: "q95", 0.99: "q99"}
LABEL = {"naive": "Naive (c x mu28 x P)", "point": "B2 (point policy, normal sigma, constant CV, pre-specified)", "point_sqrt": "B2-sqrt (point policy, normal sigma, sqrt-scale fallback)",
         "quantile": "Quantile policy (headline)", "posthoc": "B3a (point forecast + empirical residual quantiles, post-hoc)"}


def sigma_forms(cal):
    """{segment: form} from the constant-CV check on the calibration rows (Phase 9 rule)"""
    res = policy.cv_check((cal.y - cal.yhat).to_numpy(), cal.scale.to_numpy(), cal.segment.to_numpy())
    return {s: res[s]["form"] for s in SEGS}


def s_naive(tab, c, P):
    return policy.ceil_units(c * tab.mu28.to_numpy() * P)


def s_point(tab, cal, alpha, forms=None):
    forms = forms or sigma_forms(cal)
    sigma_eff = np.empty(len(tab))
    scale = tab.scale.to_numpy()
    for s in SEGS:
        m = (tab.segment == s).to_numpy()
        c = cal[cal.segment == s]
        f = policy.sigma_p((c.y - c.yhat).to_numpy(), c.scale.to_numpy(), forms[s])
        sigma_eff[m] = f(scale[m]) / scale[m]
    return policy.rop_classic(tab.yhat.to_numpy(), sigma_eff, scale, alpha)


def s_quantile(tab, alpha):
    return policy.rop_quantile(tab[QCOL[alpha]].to_numpy())


def s_posthoc(tab, cal, alpha):
    off = conformal.segment_offsets(((cal.y - cal.yhat) / cal.scale).to_numpy(), cal.segment.to_numpy(), alpha)
    return policy.ceil_units(np.maximum(tab.yhat.to_numpy() + tab.scale.to_numpy() * tab.segment.map(off).to_numpy(), 0.0))


def as_matrix(tab, values, dates, ids):
    """row vector aligned with tab -> (len(dates), len(ids)) matrix; every (date, id) must be present exactly once"""
    m = pd.DataFrame({"date": tab.date.to_numpy(), "id": tab.id.to_numpy(), "v": np.asarray(values)}).pivot(index="date", columns="id", values="v").reindex(index=dates, columns=ids)
    if m.isna().any().any():
        raise ValueError("forecast table does not cover every review date and series")
    return m.to_numpy().astype("int64")


def all_levels(tab, cal, P, c_grid, alphas=policy.SERVICE, dates=None, ids=None):
    """{(policy, setting): S matrix (K, N)} for naive over c_grid and point / quantile / posthoc over alphas, plus {segment: form} used by the point policy"""
    dates = pd.DatetimeIndex(sorted(tab.date.unique())) if dates is None else dates
    ids = sorted(tab.id.unique()) if ids is None else ids
    forms = sigma_forms(cal)
    out = {}
    for c in c_grid:
        out[("naive", c)] = as_matrix(tab, s_naive(tab, c, P), dates, ids)
    for a in alphas:
        out[("point", a)] = as_matrix(tab, s_point(tab, cal, a, forms), dates, ids)
        out[("quantile", a)] = as_matrix(tab, s_quantile(tab, a), dates, ids)
        out[("posthoc", a)] = as_matrix(tab, s_posthoc(tab, cal, a), dates, ids)
    return out, forms


def version_of(review_dates):
    """model version serving each review date, from the schedule of versions.py (warm-up dates use version 0)"""
    out = {}
    for v in versions.CUTOFFS:
        for d in versions.use_dates(v):
            out[d] = v
    return [out[pd.Timestamp(d)] for d in review_dates]
