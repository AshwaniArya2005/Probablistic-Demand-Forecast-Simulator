"""Phase 9: planner-facing quantities, exactly as pre-registered in docs/design.md section 12 (Phase 9 pre-registration): the stockout-risk label and
overstock flag from the quantile grid, classic vs quantile reorder point (= order-up-to level) and safety stock, and the constant-CV check with its
fallback. Pure functions on arrays; no data, no model. The stockout-risk label is NOT a calibrated probability of stockout (see the pre-registration)."""
import numpy as np

from quantiles import normal_z

SERVICE = (0.80, 0.90, 0.95, 0.99)
REVIEW = 7
LEADS = (3, 7)                       # lead times; protection interval P = REVIEW + L in {10, 14}
CV_BINS, CV_THRESHOLD = 4, 1.5       # constant-CV check: 4 equal-count scale bins, max/min bin RMSE at most 1.5


def protection_interval(lead):
    if lead not in LEADS:
        raise ValueError(f"lead time must be one of {LEADS}")
    return REVIEW + lead


def ceil_units(x):
    """integer ceiling; values are rounded to 1e-9 first so that arithmetic noise (12.000000000000002) does not add a unit, while a real 12.01 still
    becomes 13. Returns int64."""
    return np.ceil(np.round(np.asarray(x, float), 9)).astype("int64")


def _check_alpha(alpha):
    if alpha not in SERVICE:
        raise ValueError(f"service level must be one of {SERVICE}")


def rop_classic(yhat, sigma_seg, scale, alpha):
    """ceil(max(0, yhat + z_alpha * sigma_seg * scale)): Normal protection demand, sigma proportional to the row scale within a segment"""
    _check_alpha(alpha)
    return ceil_units(np.maximum(np.asarray(yhat, float) + normal_z(alpha) * np.asarray(sigma_seg, float) * np.asarray(scale, float), 0.0))


def rop_quantile(q):
    """ceil(q_alpha) from the sorted quantile model (the model's output is already non-negative)"""
    return ceil_units(np.maximum(np.asarray(q, float), 0.0))


def safety_stock(rop, yhat, display=True):
    """SS = ROP - yhat with the same point forecast yhat for every method; floored at 0 for display, raw value kept with display=False"""
    ss = np.asarray(rop, float) - np.asarray(yhat, float)
    return np.maximum(ss, 0.0) if display else ss


def order_quantity(rop, ip):
    """order-up-to: the reorder point is the order-up-to level S, so order max(0, S - inventory position)"""
    return np.maximum(np.asarray(rop, "int64") - np.asarray(ip, "int64"), 0)


def stockout_risk(ip, q50, q90, q99):
    """(label, overstock) with Q_a = ceil(q_a): HIGH if IP < Q50; MEDIUM if Q50 <= IP < Q90; LOW if IP >= Q90; overstock flag if IP > Q99.
    Quantiles must be ordered (q50 <= q90 <= q99); coinciding quantiles empty the bands between them (see the pre-registration)."""
    q50, q90, q99 = (np.asarray(q, float) for q in (q50, q90, q99))
    if (q50 > q90 + 1e-9).any() or (q90 > q99 + 1e-9).any():
        raise ValueError("quantiles must be sorted: q50 <= q90 <= q99")
    ip = np.asarray(ip, "int64")
    Q50, Q90, Q99 = ceil_units(q50), ceil_units(q90), ceil_units(q99)
    label = np.where(ip < Q50, "HIGH", np.where(ip < Q90, "MEDIUM", "LOW"))
    return label, ip > Q99


def _bin_rmse(resid, scale, bins):
    order = np.argsort(np.asarray(scale, float), kind="stable")
    r = np.asarray(resid, float)[order]
    return np.array([np.sqrt((chunk ** 2).mean()) for chunk in np.array_split(r, bins)])


def bin_ratio(resid, scale, bins=CV_BINS):
    """max / min of the RMSE of `resid` over `bins` equal-count bins of `scale` (inf if a bin has zero spread)"""
    rm = _bin_rmse(resid, scale, bins)
    return float(rm.max() / rm.min()) if rm.min() > 0 else float("inf")


def cv_check(error, scale, segments, bins=CV_BINS, threshold=CV_THRESHOLD):
    """constant-CV check per velocity segment. `error` = y - yhat on calibration origins. Constant CV holds if the scale-normalised residual error / scale has
    a bin ratio <= threshold; otherwise the fallback normalises by sqrt(scale) and is tested the same way. Returns {segment: dict(ratio_constant_cv,
    ratio_sqrt_scale, form, n)} with form in {"constant-CV", "sqrt-scale", "constant-CV (both failed)"}."""
    e, sc, sg = np.asarray(error, float), np.asarray(scale, float), np.asarray(segments)
    out = {}
    for s in np.unique(sg):
        m = sg == s
        r1, r2 = bin_ratio(e[m] / sc[m], sc[m], bins), bin_ratio(e[m] / np.sqrt(sc[m]), sc[m], bins)
        form = "constant-CV" if r1 <= threshold else "sqrt-scale" if r2 <= threshold else "constant-CV (both failed)"
        out[s] = dict(ratio_constant_cv=r1, ratio_sqrt_scale=r2, form=form, n=int(m.sum()))
    return out


def sigma_p(error, scale, form):
    """pooled sigma for one segment under the chosen form: RMSE of error/scale (constant CV, sigma_P = sigma * scale) or of error/sqrt(scale)
    (sigma_P = sigma' * sqrt(scale)). Returns a function scale -> sigma_P."""
    e, sc = np.asarray(error, float), np.asarray(scale, float)
    if form == "sqrt-scale":
        s = float(np.sqrt(((e / np.sqrt(sc)) ** 2).mean()))
        return lambda scl: s * np.sqrt(np.asarray(scl, float))
    s = float(np.sqrt(((e / sc) ** 2).mean()))
    return lambda scl: s * np.asarray(scl, float)
