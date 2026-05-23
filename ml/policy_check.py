"""Phase 9 on the tuning folds (report-only, definitions fixed in docs/design.md section 12): (1) the constant-CV check per velocity segment, (2) classic vs
quantile (and the post-hoc empirical-residual) reorder points and safety stock, (3) a validity check of the stockout-risk labels. Needs the parquet files of
ml/policy_rows.py. Writes docs/results/phase9_policy_quantities_tuning.md. Run: uv run python ml/policy_check.py"""
import glob
import subprocess
from datetime import date

import numpy as np
import pandas as pd

import conformal
import policy
from config import PROCESSED, ROOT

E = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase9_eval_*.parquet")))], ignore_index=True)
C = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase9_cal_*.parquet")))], ignore_index=True)
assert E[["fold", "P"]].drop_duplicates().shape[0] == 12 and (pd.DatetimeIndex(E.date).dayofweek == 6).all(), "12 cells of Sunday review origins expected"
SEGS = ("low", "mid", "high")
QCOL = {0.80: "q80", 0.90: "q90", 0.95: "q95", 0.99: "q99"}
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

# ---- 1. constant-CV check on the calibration windows (never on the evaluation origins) ----
cv_rows = []
form = {}
for (fold, P), g in C.groupby(["fold", "P"]):
    res = policy.cv_check((g.y - g.yhat).to_numpy(), g.scale.to_numpy(), g.segment.to_numpy())
    for s, r in res.items():
        cv_rows.append(dict(fold=fold, P=P, segment=s, **r))
        form[(fold, P, s)] = r["form"]
cv = pd.DataFrame(cv_rows)
cv_sum = cv.groupby(["segment", "form"]).size().unstack(fill_value=0).reindex(SEGS)
ratios = cv.groupby("segment")[["ratio_constant_cv", "ratio_sqrt_scale"]].agg(["median", "max"]).reindex(SEGS)

# ---- 2. classic vs quantile vs post-hoc empirical reorder points, P in {10, 14} ----
rows = []
for (fold, P), g in E[E.P.isin([10, 14])].groupby(["fold", "P"]):
    cal = C[(C.fold == fold) & (C.P == P)]
    for a in policy.SERVICE:
        sc_cal = ((cal.y - cal.yhat) / cal.scale).to_numpy()
        off = conformal.segment_offsets(sc_cal, cal.segment.to_numpy(), a)
        sigma_eff = np.empty(len(g))
        for s in SEGS:
            m = (g.segment == s).to_numpy()
            c = cal[cal.segment == s]
            f = policy.sigma_p((c.y - c.yhat).to_numpy(), c.scale.to_numpy(), form[(fold, P, s)])
            sigma_eff[m] = f(g.scale.to_numpy()[m]) / g.scale.to_numpy()[m]
        yhat, scale, y = g.yhat.to_numpy(), g.scale.to_numpy(), g.y.to_numpy()
        for name, rop in (("classic (normal)", policy.rop_classic(yhat, sigma_eff, scale, a)), ("quantile", policy.rop_quantile(g[QCOL[a]].to_numpy())),
                          ("post-hoc empirical residual", policy.ceil_units(np.maximum(yhat + scale * g.segment.map(off).to_numpy(), 0.0)))):
            rows.append(pd.DataFrame({"fold": fold, "P": P, "alpha": a, "method": name, "segment": g.segment.to_numpy(), "rop": rop,
                                      "ss": policy.safety_stock(rop, yhat), "covered": y <= rop}))
R = pd.concat(rows, ignore_index=True)
cmp_all = R.groupby(["alpha", "method"]).agg(mean_rop=("rop", "mean"), mean_safety_stock=("ss", "mean"), coverage=("covered", "mean")).round(3)
cmp_seg = R.pivot_table(index=["alpha", "segment"], columns="method", values="ss", aggfunc="mean").round(2).reindex(SEGS, level=1)

# ---- 3. label validity: reference positions IP = ceil(c * mu28 * P), P in {10, 14} ----
lab = []
for c in (1.0, 1.5, 2.0):
    g = E[E.P.isin([10, 14])]
    ip = policy.ceil_units(c * g.mu28.to_numpy() * g.P.to_numpy())
    label, over = policy.stockout_risk(ip, g.q50.to_numpy(), g.q90.to_numpy(), g.q99.to_numpy())
    short = g.y.to_numpy() > ip
    for L in ("HIGH", "MEDIUM", "LOW"):
        m = label == L
        lab.append(dict(c=c, label=L, share_of_origins=m.mean(), n=int(m.sum()), realised_stockout_freq=short[m].mean() if m.any() else np.nan))
    lab.append(dict(c=c, label="overstock flag", share_of_origins=over.mean(), n=int(over.sum()), realised_stockout_freq=short[over].mean() if over.any() else np.nan))
LB = pd.DataFrame(lab).round(3)

md = f"""# Phase 9 planner quantities on the tuning folds

Run {date.today()} at code commit `{head}`. Report-only; **tuning folds only**, touch log unchanged. Definitions, thresholds and the constant-CV rule were fixed in docs/design.md
(Phase 9 pre-registration) before this run. Rows: {len(E)} series-origins (12 fold x horizon cells of Sunday review origins); reorder points use P in {{10, 14}}.

## 1. Constant-CV check (calibration windows; 4 scale bins per segment; accepted if max/min bin RMSE <= 1.5, else the sqrt(scale) fallback)

Count of fold x horizon cells (12 per segment) by the form the check selects:

{cv_sum.to_markdown()}

Bin ratio of scale-normalised residuals (constant-CV form) and of the fallback, by segment (median and max over the 12 cells):

{ratios.round(2).to_markdown()}

## 2. Classic vs quantile reorder points (pooled over folds, P = 10 and 14; coverage = share of actual protection demand at or below the reorder point)

{cmp_all.to_markdown()}

Mean safety stock (units) by segment:

{cmp_seg.to_markdown()}

## 3. Stockout-risk label validity (reference stock positions IP = ceil(c x mu28 x P), c = 1, 1.5, 2; not a policy, only a check of what the labels mean)

{LB.to_markdown(index=False)}

Reading: the label is a description of the position against the model's demand distribution, not a calibrated probability (design.md, Phase 9 pre-registration, item 1).
"""
(ROOT / "docs" / "results" / "phase9_policy_quantities_tuning.md").write_text(md, encoding="utf-8")
cv.round(4).to_csv(ROOT / "docs" / "results" / "phase9_constant_cv_cells.csv", index=False)
print(md)
