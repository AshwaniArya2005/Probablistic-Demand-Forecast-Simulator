"""Report-only (design.md section 12, 2026-09-21): for the frozen Phase 7 quantile model against each benchmark, the number of the 12 fold x horizon cells
it wins on mean scaled pinball, and an item-cluster bootstrap 95% interval on the difference (quantile model minus benchmark; negative is better).
Items (100), not series (300), are resampled: an item's three store-series share its demand drivers. 10,000 resamples, seed 0. The interval reflects item
sampling only, not period-to-period variability (one set of folds). Needs data/processed/phase7_series_*.parquet from ml/series_pinball.py.
Run: uv run python ml/bootstrap_quantile.py"""
import glob
import json

import numpy as np
import pandas as pd

from config import PROCESSED, ROOT

B, SEED = 10000, 0
BENCH = {"B1_normal": "B1: MA-28 + normal sigma", "B2_normal": "B2: XGBoost mean + normal sigma (point policy)",
         "B2raw_normal": "B2': frozen Phase 6 raw XGBoost + normal sigma", "B3a_xgb": "B3a (post-hoc): XGBoost mean + empirical residual quantiles",
         "B3b_raw_xgb": "B3b (post-hoc): frozen raw XGBoost + empirical residual quantiles", "B3c_ma28": "B3c (post-hoc): MA-28 + empirical residual quantiles"}

D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase7_series_*.parquet")))], ignore_index=True)
assert D.groupby("method").size().nunique() == 1 and D[["fold", "P"]].drop_duplicates().shape[0] == 12
items = np.sort(D.item.unique())
cells = D[["fold", "P"]].drop_duplicates().sort_values(["fold", "P"]).itertuples(index=False, name=None)
cells = list(cells)
rng = np.random.default_rng(SEED)
W = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=B).astype("float64")
q = D[D.method == "quantile_model"].set_index(["fold", "P", "id"])


def ci(diff, sel_cells):
    """diff: DataFrame with fold, P, item and d = quantile model minus benchmark, restricted to the rows of a group; statistic = mean over cells of the per-cell mean"""
    S = np.zeros((len(items), len(sel_cells)))
    N = np.zeros_like(S)
    pos = {c: j for j, c in enumerate(sel_cells)}
    ix = {it: i for i, it in enumerate(items)}
    for (f, p, it), g in diff.groupby(["fold", "P", "item"]):
        S[ix[it], pos[(f, p)]], N[ix[it], pos[(f, p)]] = g.d.sum(), len(g)
    est = np.mean(S.sum(0) / N.sum(0))
    boot = np.mean((W @ S) / (W @ N), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


out = []
for m, label in BENCH.items():
    b = D[D.method == m].set_index(["fold", "P", "id"])
    d = (q.sp_mean6 - b.sp_mean6).dropna().rename("d").reset_index()
    meta = D.drop_duplicates("id").set_index("id")
    d["item"], d["segment"] = d.id.map(meta.item), d.id.map(meta.segment)
    per_cell = d.groupby(["fold", "P"]).d.mean()
    won = int((per_cell < 0).sum())
    won_f2 = int((per_cell.xs("F2", level="fold") < 0).sum())
    groups = {"all 12 cells": (d, cells), "F2 (3 cells)": (d[d.fold == "F2"], [c for c in cells if c[0] == "F2"]),
              **{f"P = {P} (4 folds)": (d[d.P == P], [c for c in cells if c[1] == P]) for P in (7, 10, 14)},
              **{f"segment {s} (12 cells)": (d[d.segment == s], cells) for s in ("low", "mid", "high")}}
    for name, (g, cs) in groups.items():
        est, lo, hi = ci(g, cs)
        out.append(dict(benchmark=label, cells_won_of_12=won, F2_cells_won_of_3=won_f2, group=name, diff=est, ci_low=lo, ci_high=hi,
                        excludes_zero="yes" if hi < 0 or lo > 0 else "no"))
res = pd.DataFrame(out)
res.round(4).to_csv(ROOT / "docs" / "results" / "phase7_bootstrap.csv", index=False)

# B3 summary (mean / median over series per cell, averaged over cells) and coverage status counts
b3 = {}
for f in sorted(glob.glob(str(PROCESSED / "phase7_b3_*.json"))):
    b3.update(json.load(open(f)))
summ = D.groupby(["method", "fold", "P", "id"]).sp_mean6.mean().groupby(["method", "fold", "P"]).agg(["mean", "median"]).groupby("method").mean()
summ = summ.loc[["quantile_model", *BENCH]]
cov = []
for label in [k for k in BENCH if k.startswith("B3")]:
    recs = [r for k, r in b3.items() if k.startswith(label + "|")]
    for a in (0.8, 0.9, 0.95, 0.99):
        st = pd.Series([r[f"status_{a}"] for r in recs]).value_counts().to_dict()
        cov.append(dict(method=label, alpha=a, cov_hi=round(np.mean([r[f"cov_hi_{a}"] for r in recs]), 3), cov_lo=round(np.mean([r[f"cov_lo_{a}"] for r in recs]), 3),
                        under=st.get("under", 0), consistent=st.get("consistent", 0), over=st.get("over", 0)))
md = ("# Phase 7: cells won, item-cluster bootstrap and the post-hoc B3 benchmarks\n\nReport-only, tuning folds only (touch log unchanged). "
      "The quantile model is the frozen Phase 7 model (normalised target, floor 1/7, depth 6 / min_child_weight 30, base feature set); per-series values were "
      "recomputed and reproduce the cached Phase 7 cell values to 1e-9. `diff` = mean scaled pinball of the quantile model minus the benchmark (negative = quantile model better), "
      f"95% percentile interval from {B} resamples of the 100 items (each item's three store-series together, seed 0); it reflects item sampling only, not "
      "period-to-period variability. `cells_won_of_12` counts fold x horizon cells where the quantile model's mean scaled pinball is lower. "
      "**Framing rule (design.md, 2026-09-21):** the ratio to B2 below is never quoted alone. Read the headline as: vs the post-hoc B3a benchmark, about 13% lower scaled pinball on the tuning folds; the normal sigma was most of the gap to the textbook policy B2. "
      "**B3 variants are post-hoc**: defined after the Phase 7 results were known (design.md section 12, 2026-09-21), they can change no decision.\n\n"
      "## Mean scaled pinball by method (mean over 12 cells of the per-cell mean and median over series)\n\n" + summ.round(4).to_markdown()
      + "\n\n## Coverage of ceil(q) for the post-hoc B3 methods (mean over cells; status counts over 12 cells)\n\n" + pd.DataFrame(cov).to_markdown(index=False)
      + "\n\n## Cells won and bootstrap intervals\n\n" + res.round(4).to_markdown(index=False) + "\n")
(ROOT / "docs" / "results" / "phase7_bootstrap.md").write_text(md, encoding="utf-8")
print(md)
