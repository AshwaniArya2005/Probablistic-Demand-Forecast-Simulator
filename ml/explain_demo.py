"""Phase 8 on real data, tuning folds only: fit the frozen Phase 7 final quantile model for one fold and horizon, explain the P50 and the P90 order-up-to
quantile of every Sunday review origin of that fold, verify additivity on every row, and write docs/results/phase8_explanations_tuning.md.
No test-window origin is explained (explain_rows refuses them). Run: XGB_THREADS=5 uv run python ml/explain_demo.py [fold] [P] [alpha]"""
import sys
from datetime import date

import numpy as np
import pandas as pd

import explain
import tune_quantile as tq
from learned import XGBQ

FOLD, P, ALPHA = (sys.argv[2] if len(sys.argv) > 2 else "F2"), int(sys.argv[3]) if len(sys.argv) > 3 else 14, float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
floor = tq.chosen_floor()
fit, sel, cal, nscale = tq.cell_data(FOLD, P)
model = XGBQ(P, seed=0, columns=tq.cols_for(P, "norm", "base"), normalize=True, floor=floor, nthread=tq.NTHREAD,
             max_depth=6, min_child_weight=30).fit(fit, fit[f"y_p{P}"])
rows = sel.copy()
res = explain.explain_rows(model, rows, P, ALPHA)                      # raises for non-Sunday or test-window origins
c = model.contributions(rows, [0.50, ALPHA])
raw = model._raw(rows)[:, [model.ALPHAS.index(0.50), model.ALPHAS.index(ALPHA)]]
gap = np.abs(c["value"] - raw)
rel = gap / np.maximum(np.abs(raw), 1.0)
own = np.array([model.ALPHAS.index(0.50), model.ALPHAS.index(ALPHA)])
crossed = (c["source"] != own[None, :])
theme_units, themes = explain.theme_contributions(c["phi"], c["names"])
seg = rows.id.map(tq.seg).to_numpy()

tab = []
for k, label in enumerate(("P50", f"P{int(ALPHA * 100)}")):
    for s in ("all", "low", "mid", "high"):
        m = np.ones(len(rows), bool) if s == "all" else seg == s
        a = np.abs(theme_units[m, k]).mean(0)
        tab.append(dict(quantile=label, segment=s, **{t: a[j] / a.sum() for j, t in enumerate(themes)}, mean_abs_total_units=a.sum(), mean_baseline_units=c["bias"][m, k].mean(),
                        mean_forecast_units=c["value"][m, k].mean()))
T = pd.DataFrame(tab)

# examples by a fixed rule: per segment the row nearest the segment median of the P50 forecast and the one nearest its 90th percentile, plus the row
# with the largest absolute price contribution to the P90 and one crossing row if any
ex = []
for s in ("high", "mid", "low"):
    v = c["value"][:, 0]
    idx = np.flatnonzero(seg == s)
    for q in (0.5, 0.9):
        ex.append((f"{s} velocity, P50 forecast near the segment's {int(q * 100)}th percentile", idx[np.argmin(np.abs(v[idx] - np.quantile(v[idx], q)))]))
pj = themes.index("price vs usual")
ex.append(("largest price-vs-usual contribution to the service-level quantile", int(np.argmax(np.abs(theme_units[:, 1, pj])))))
if crossed.any():
    ex.append(("a row where XGBoost's internal sort changed the quantile order", int(np.flatnonzero(crossed.any(axis=1))[0])))
head = f"`{tq.subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=tq.ROOT).stdout.strip()}`"
md = [f"# Phase 8 explanations on a tuning fold ({FOLD}, P = {P}, service-level quantile {ALPHA})\n\nRun {date.today()} at code commit {head}. **Tuning fold only**; no test-window "
      "origin is explained (`explain_rows` refuses them) and the touch log is unchanged. Model: the frozen Phase 7 final quantile model (normalised target, floor 1/7, depth 6 / "
      "min_child_weight 30, base feature set) fitted for this fold's version. Contributions are exact TreeSHAP values in units (ratio-space contributions times each row's scale) "
      "of the reported quantile; they describe **associations in the model, not causal effects**. Explained rows: every Sunday review origin of the fold "
      f"({len(rows)} series-origins).\n",
      "## Exactness checks over every explained row\n",
      f"- Additivity (baseline + sum of contributions = the reported quantile before clipping at 0): max absolute gap {gap.max():.2e} units, max relative gap {rel.max():.2e}, "
      f"for the P50 and the P{int(ALPHA * 100)} together ({gap.size} values).",
      f"- Rows where the explained quantile came from another target's trees because XGBoost's internal sort changed the order: {int(crossed.any(axis=1).sum())} of {len(rows)} "
      f"({crossed.any(axis=1).mean():.3%}); the fit's unsorted crossing share is {model.crossing_share(rows):.3%}.",
      f"- Rows whose reported value is below 0 (floored at 0 in the forecast): {int((c['value'] < 0).any(axis=1).sum())}.\n",
      "## Share of the absolute contribution by theme (mean over rows)\n", T.round(3).to_markdown(index=False) + "\n",
      "## Example explanations (chosen by a fixed rule, not by appeal)\n"]
for why, i in ex:
    r = res[i]
    row = rows.iloc[i]
    md.append(f"**{why}** (`{row.id}`, origin {row.date.date()}, velocity segment {seg[i]}):\n\n- {r['p50']['text']}\n- {r['service']['text']}\n")
out = tq.ROOT / "docs" / "results" / "phase8_explanations_tuning.md"
out.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
