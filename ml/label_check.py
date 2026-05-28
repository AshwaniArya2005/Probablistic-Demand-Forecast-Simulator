"""Phase 9 follow-up (report-only, tuning folds): why is the realised shortfall frequency among HIGH rows below one half? Splits HIGH rows by the ceiled
median Q50, by the gap Q50 - IP, by whether the item was in a zero run at the origin (zero_run_91 >= 14, the outage marker of design.md) and by the tie
mass. Needs the parquet files of ml/policy_rows.py. Writes docs/results/phase9_label_investigation_tuning.md. Run: uv run python ml/label_check.py"""
import glob
import subprocess
from datetime import date

import numpy as np
import pandas as pd

import policy
from config import PROCESSED, ROOT

E = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase9_eval_*.parquet")))], ignore_index=True)
F = pd.read_parquet(PROCESSED / "features.parquet", columns=["id", "date", "zero_run_91", "zero_frac_28"])
E = E[E.P.isin([10, 14])].merge(F, on=["id", "date"], how="left")
assert E.zero_run_91.notna().all()
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
Q50, Q90 = policy.ceil_units(E.q50), policy.ceil_units(E.q90)
out = []
for c in (1.0, 1.5, 2.0):
    ip = policy.ceil_units(c * E.mu28.to_numpy() * E.P.to_numpy())
    label, _ = policy.stockout_risk(ip, E.q50.to_numpy(), E.q90.to_numpy(), E.q99.to_numpy())
    d = E.assign(c=c, IP=ip, Q50=Q50, gap=Q50 - ip, short=E.y.to_numpy() > ip, label=label, ge_q50=E.y.to_numpy() >= Q50)
    out.append(d[d.label == "HIGH"])
H = pd.concat(out, ignore_index=True)
H["q50_band"] = pd.cut(H.q50, [0, 1, 2, 5, np.inf], labels=["0 < q50 <= 1 (Q50 = 1)", "1 < q50 <= 2 (Q50 = 2)", "2 < q50 <= 5", "q50 > 5"])
H["gap_band"] = pd.cut(H.gap, [0, 1, 3, 10, np.inf], labels=["gap 1", "gap 2-3", "gap 4-10", "gap > 10"])
H["outage"] = np.where(H.zero_run_91 >= 14, "in a zero run of 14+ days", "no long zero run")


def table(by):
    return H.groupby(["c", by], observed=True).agg(n=("short", "size"), share_of_HIGH=("short", lambda s: len(s)), realised_shortfall=("short", "mean"),
                                                  y_at_or_above_Q50=("ge_q50", "mean")).round(3).drop(columns="share_of_HIGH")


# unconditional check of the median itself: P(y >= Q50) over ALL rows, and where q50 > 0
allrows = E.assign(Q50=Q50, ge=E.y.to_numpy() >= Q50)
uncond = allrows[allrows.q50 > 0].groupby("segment").agg(n=("ge", "size"), y_at_or_above_Q50=("ge", "mean")).round(3)
zero_mass = E.groupby(pd.cut(E.q50, [-1e-9, 0, 0.5, 1, 2, 5, np.inf], labels=["q50 = 0", "0-0.5", "0.5-1", "1-2", "2-5", ">5"]), observed=True).agg(
    n=("y", "size"), share_y_zero=("y", lambda s: (s == 0).mean()), share_y_at_or_above_Q50=("y", lambda s: 0)).round(3).drop(columns="share_y_at_or_above_Q50")

md = f"""# Phase 9 follow-up: why realised shortfall among HIGH rows is below one half

Run {date.today()} at code commit `{head}`. Report-only, tuning folds only, touch log unchanged. Label definition unchanged (design.md, Phase 9 pre-registration).
For a calibrated integer model, IP < Q50 implies P(demand > IP) >= P(demand >= Q50) >= 0.5, so realised shortfall well below one half in HIGH rows means the
model's median is too high **on those rows**, or the rows are not what the label assumes. HIGH rows are pooled over the c = 1, 1.5, 2 reference positions
(P in {{10, 14}}); `y_at_or_above_Q50` is the share of those rows whose demand reached the ceiled median.

## Realised shortfall among HIGH rows by the median

{table("q50_band").to_markdown()}

## by the gap Q50 - IP (units the reference position sits below the ceiled median)

{table("gap_band").to_markdown()}

## by outage status at the origin (zero run of 14 or more days in the last 91)

{table("outage").to_markdown()}

## by velocity segment

{table("segment").to_markdown()}

## The median itself, all origins with q50 > 0: share with demand at or above Q50 (should be about 0.5 or more)

{uncond.to_markdown()}

## Share of origins with zero demand, by the size of q50 (all rows, P in {{10, 14}})

{zero_mass.to_markdown()}
"""
(ROOT / "docs" / "results" / "phase9_label_investigation_tuning.md").write_text(md, encoding="utf-8")
print(md)
