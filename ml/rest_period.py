"""Why did every policy serve above its target in the rest period, more than in the peak? (touch-log row "Phase 10 diagnostic restperiod"; hypotheses in docs/design.md, written before this ran.)
For the base-case test-window replay it prints, by period and at each of the four service levels, for the quantile policy: (a) the P-day coverage of ceil(q) at the review origins, (b) the simulated
cycle service and the wedge between them, (c) the same excluding series-cycles with zero demand over the protection interval, (d) the share of zero-demand series-cycles, (e) realised demand against
the forecasts (sum y / sum yhat and sum y / MA-28 total). Aggregates only; no per-day data is written. Run: uv run python ml/rest_period.py"""
from datetime import date

import numpy as np
import pandas as pd

import sim_run as sr
import simulator as sim
from config import ROOT
from folds import touch_logged

if not touch_logged("Phase 10 diagnostic restperiod"):
    raise SystemExit("refusing to run: no 'Phase 10 diagnostic restperiod' row in the touch log (docs/design.md section 14)")
run = sr.test_runs()[0]
ids = sorted(run.tab.id.unique())
dem, _, _, _ = sr.load_market(ids)
levels, _ = sr.levels_for(run, (1.0,), ids)
days = dem.index
rv = np.array([days.get_loc(d) for d in run.dates])
mk = np.arange(run.n_warmup, len(rv))
L, P = run.L, run.P
D = dem.to_numpy()
peak = sr.peak_mask(run)
tab = run.tab.set_index(["date", "id"])
dates = list(run.dates[run.n_warmup:])
yP = np.array([[D[rv[k] + 1:rv[k] + P + 1, j].sum() for j in range(len(ids))] for k in mk], float)               # P-day demand per (counted review, series)
yhat = np.array([[tab.loc[(d, i), "yhat"] for i in ids] for d in dates], float)
ma = np.array([[tab.loc[(d, i), "mu28"] * P for i in ids] for d in dates], float)
rows = []
for a in sr.ALPHAS:
    S = levels[("quantile", a)]
    rep = sim.replay(D, rv, S, L)
    t = sim.cycle_table(rep, D, rv, L, mk, P=P)
    Sm = S[mk].astype(float)
    for per, m in (("peak", peak), ("rest", ~peak), ("all", np.ones(len(peak), bool))):
        cov = (yP[m] <= Sm[m]).mean()
        cs = 1 - t["stockout"][m].mean()
        pos = t["zero"][m] == 0
        cs_pos = 1 - t["stockout"][m][pos].mean()
        rows.append(dict(alpha=a, period=per, review_dates=int(m.sum()), coverage_of_ceil_q=f"{cov:.4f}", cycle_service=f"{cs:.4f}", wedge=f"{cs - cov:+.4f}", cycle_service_positive_demand_only=f"{cs_pos:.4f}",
                         zero_demand_share=f"{t['zero'][m].mean():.4f}"))
sw = pd.DataFrame(rows)
lvl = pd.DataFrame([{"period": per, "sum y / sum yhat (XGBoost mean)": f"{yP[m].sum() / yhat[m].sum():.4f}", "sum y / sum MA-28": f"{yP[m].sum() / ma[m].sum():.4f}", "mean P-day demand per series": f"{yP[m].mean():.3f}",
                    "zero-demand share of series-origins": f"{(yP[m] == 0).mean():.4f}"} for per, m in (("peak", peak), ("rest", ~peak), ("all", np.ones(len(peak), bool)))])
md = f"""# Rest-period service: diagnostic (test window, base case, quantile policy)

Run {date.today()}. Peak = cycles starting on or before 2016-01-03 ({int(peak.sum())} reviews), rest = the other {int((~peak).sum())}. `coverage_of_ceil_q` = share of review origins whose realised P-day demand was at or below the
order-up-to level ceil(q); `cycle_service` = share of series-cycles without a stockout in the simulation; `wedge` = their difference; `positive_demand_only` drops series-cycles whose protection-interval demand was zero.

{sw.to_markdown(index=False)}

## Realised demand against the forecasts

{lvl.to_markdown(index=False)}
"""
(ROOT / "docs" / "results" / "phase10_restperiod_diagnostic.md").write_text(md, encoding="utf-8")
print(md)
