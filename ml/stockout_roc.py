"""Stockout-risk ROC/PR diagnostic (design.md, 2026-09-22 pre-registration item 1). Tuning folds only, exploratory, no retraining, no new model, no change to
any policy or label. Ground truth and score are fixed exactly as pre-registered:
  - reference positions IP = ceil(c * mu28 * P), c in {1.0, 1.5, 2.0}, P in {10, 14} (data/processed/phase9_eval_*.parquet, the frozen Phase 9 rows)
  - ground truth: shortfall = 1 if y > IP else 0
  - score: 1 - CDF_hat(IP), CDF_hat a monotone piecewise-linear interpolation through (0, 0), (0.10, q10), (0.50, q50), (0.80, q80), (0.90, q90),
    (0.95, q95), (0.99, q99); above q99 the last segment's slope is extended, capped so the interpolated level never exceeds 0.999 (a ties-flat segment
    maps every IP inside it to its lower endpoint's alpha, i.e. frac = 0 there).
Results are reported exactly as computed (the pre-registration rule): this script cannot change the frozen model, the label cut points, or any policy.
Run: uv run python ml/stockout_roc.py"""
import glob
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import policy
from config import PROCESSED, ROOT

C_GRID = (1.0, 1.5, 2.0)
ALPHAS_ANCHOR = np.array([0.0, 0.10, 0.50, 0.80, 0.90, 0.95, 0.99])
QCOLS = ["q10", "q50", "q80", "q90", "q95", "q99"]


def implied_score(ip, Y):
    """1 - CDF_hat(ip) by piecewise-linear interpolation through (ALPHAS_ANCHOR, Y) per row; Y = (n, 7) with column 0 all zeros. See module docstring."""
    ip = np.asarray(ip, float)
    n = len(ip)
    idx_arange = np.arange(n)
    q99 = Y[:, 6]
    beyond = ip > q99                                     # strictly past q99: the extended-slope branch
    # first anchor >= ip = count of anchors strictly < ip; the segment is the one just before it. Using a STRICT count (not <=) means a tie
    # (e.g. q80 == q90) resolves to the segment ending at the tie's first occurrence, i.e. the tie's lower alpha (checked by hand in ml/stockout_roc.py's
    # module test snippet: q80 = q90 = 8 with ip = 8 gives alpha 0.80, not 0.90).
    j = (Y < ip[:, None]).sum(1)
    idx = np.clip(j - 1, 0, 5)
    y0, y1 = Y[idx_arange, idx], Y[idx_arange, idx + 1]
    x0, x1 = ALPHAS_ANCHOR[idx], ALPHAS_ANCHOR[idx + 1]
    denom = y1 - y0
    frac = np.clip(np.where(denom > 0, (ip - y0) / np.where(denom > 0, denom, 1.0), 0.0), 0.0, 1.0)
    alpha = x0 + frac * (x1 - x0)
    q95 = Y[:, 5]
    last_dx = ALPHAS_ANCHOR[6] - ALPHAS_ANCHOR[5]                    # 0.04: the (0.95, 0.99) segment whose slope is extended past q99
    slope = np.where(q99 > q95, last_dx / np.where(q99 > q95, q99 - q95, 1.0), 0.0)
    alpha_ext = np.minimum(0.999, 0.99 + slope * (ip - q99))
    alpha = np.where(beyond, alpha_ext, alpha)
    return 1.0 - alpha


def curve_points(fn, y_true, score, n=40):
    """downsample a (x, y, ...) curve from sklearn to n evenly-spaced points for the saved CSV; the AUC/AP numbers use the full-resolution curve"""
    out = fn(y_true, score)
    x, y = out[0], out[1]
    if len(x) > n:
        pick = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
        x, y = x[pick], y[pick]
    return x, y


if __name__ == "__main__":
    E = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase9_eval_*.parquet")))], ignore_index=True)
    E = E[E.P.isin([10, 14])].reset_index(drop=True)
    Y_all = np.column_stack([np.zeros(len(E))] + [E[c].to_numpy(float) for c in QCOLS])

    rows, curves = [], []
    for c in C_GRID:
        ip = policy.ceil_units(c * E.mu28.to_numpy() * E.P.to_numpy()).astype(float)
        y_true = (E.y.to_numpy() > ip).astype(int)
        score = implied_score(ip, Y_all)
        for label, mask in [("pooled", np.ones(len(E), bool))] + [(f"P={p}", (E.P == p).to_numpy()) for p in (10, 14)]:
            yt, sc = y_true[mask], score[mask]
            if yt.min() == yt.max():
                rows.append(dict(c=c, slice=label, n=int(mask.sum()), positive_rate=float(yt.mean()), roc_auc=float("nan"), average_precision=float("nan")))
                continue
            roc_auc, ap = roc_auc_score(yt, sc), average_precision_score(yt, sc)
            rows.append(dict(c=c, slice=label, n=int(mask.sum()), positive_rate=round(float(yt.mean()), 4), roc_auc=round(roc_auc, 4), average_precision=round(ap, 4)))
            if label == "pooled":
                fpr, tpr = curve_points(roc_curve, yt, sc)
                for x, y in zip(fpr, tpr):
                    curves.append(dict(c=c, curve="roc", x=round(float(x), 4), y=round(float(y), 4)))
                prec, rec = curve_points(precision_recall_curve, yt, sc)
                for x, y in zip(rec, prec):
                    curves.append(dict(c=c, curve="pr", x=round(float(x), 4), y=round(float(y), 4)))
    R = pd.DataFrame(rows)
    C = pd.DataFrame(curves)
    C.to_csv(ROOT / "docs" / "results" / "phase11_stockout_roc_curves.csv", index=False)

    md = f"""# Stockout-risk ROC/PR diagnostic (tuning folds, exploratory; design.md 2026-09-22 pre-registration)

Run {date.today()}. Ground truth, reference positions and the interpolated score are fixed by the pre-registration (not tuned here). No retraining; the frozen quantile model's own six quantiles
(q10, q50, q80, q90, q95, q99) are the only inputs to the score. Reported as computed, including any weak slice.

{R.to_markdown(index=False)}

Full-resolution ROC and precision-recall curve points (downsampled to at most 40 points per curve for size): `docs/results/phase11_stockout_roc_curves.csv`.

Reading: this is a diagnostic of how well the model's existing quantile grid discriminates shortfall when read as a continuous score; it does not change the stockout-risk label's cut points
(P50 / P90 / P99, design.md Phase 9 pre-registration) or any policy, and a low AUC in a slice is reported as such, not adjusted for.
"""
    (ROOT / "docs" / "results" / "phase11_stockout_roc_tuning.md").write_text(md, encoding="utf-8")
    print(md)
