"""Phase 10 diagnostic (touch-log row "Phase 10 diagnostic leadcheck"): the quantile policy's unrounded fill rate at every alpha under the fixed lead time and under each of the ten random-lead
draws, confirmation that the draws were applied (realised leads, orders whose arrival day moved, on-hand paths that changed), and how much of the unmet demand at alpha 0.99 sits in cycles whose
order-up-to level was zero (a structural floor no lead time can lift). Test window, read-only, no result file is overwritten. Run: uv run python ml/lead_check.py"""
import numpy as np

import sim_run as sr
import simulator as sim
from folds import touch_logged

if not touch_logged("Phase 10 diagnostic leadcheck"):
    raise SystemExit("refusing to run: no 'Phase 10 diagnostic leadcheck' row in the touch log (docs/design.md section 14)")
run = sr.test_runs()[0]
ids = sorted(run.tab.id.unique())
dem, price, item_of_id, seg = sr.load_market(ids)
levels, _ = sr.levels_for(run, (1.0,), ids)
days = dem.index
rv = np.array([days.get_loc(d) for d in run.dates])
mk = np.arange(run.n_warmup, len(rv))
L = run.L
D = dem.to_numpy()


def cyc(rep, S):
    t = sim.cycle_table(rep, D, rv, L, mk)
    zero_S = (S[mk] == 0)
    return t, zero_S


print(f"{'alpha':>6} {'lead':>10} {'fill':>10} {'lost units':>11} {'demanded':>10} {'cycle service':>14}")
draws = sr.lead_draws()
for a in sr.ALPHAS:
    S = levels[("quantile", a)]
    t, zS = cyc(sim.replay(D, rv, S, L), S)
    print(f"{a:>6} {'fixed 3':>10} {1 - t['lost'].sum() / t['demanded'].sum():>10.6f} {int(t['lost'].sum()):>11} {int(t['demanded'].sum()):>10} {1 - t['stockout'].mean():>14.6f}")
    fills, lost, css = [], [], []
    for s, lead in draws:
        tt, _ = cyc(sim.replay(D, rv, S, L, lead=lead), S)
        fills.append(1 - tt["lost"].sum() / tt["demanded"].sum())
        lost.append(int(tt["lost"].sum()))
        css.append(1 - tt["stockout"].mean())
    print(f"{a:>6} {'random x10':>10} {np.mean(fills):>10.6f} {np.mean(lost):>11.1f} {int(tt['demanded'].sum()):>10} {np.mean(css):>14.6f}   per-draw fill min {min(fills):.6f} max {max(fills):.6f}")

# confirmation that the draws were applied
K, N = levels[("quantile", 0.99)].shape
lead0 = draws[0][1]
print("\nlead draw 0: shape", lead0.shape, "share of leads equal to 2 / 3 / 4:", [round(float((lead0 == v).mean()), 3) for v in (2, 3, 4)], "mean", round(float(lead0.mean()), 3))
S = levels[("quantile", 0.99)]
fixed, rnd = sim.replay(D, rv, S, L), sim.replay(D, rv, S, L, lead=lead0)
print("orders placed (review > 0, quantity > 0):", int((fixed.orders[1:] > 0).sum()), "; orders whose realised lead differs from 3:", int(((fixed.orders[1:] > 0) & (lead0[1:] != 3)).sum()))
print("series-days whose end-of-day on-hand differs, fixed vs random draw 0:", int((fixed.onhand != rnd.onhand).sum()), "of", int(fixed.onhand.size), "; order quantities differ in", int((fixed.orders != rnd.orders).sum()), "of", int(fixed.orders.size), "cells")

# structural floor at alpha 0.99: unmet demand in cycles whose own order-up-to level was zero
t, zS = cyc(fixed, S)
lost_zero = int((t["lost"] * zS).sum())
print(f"\nalpha 0.99, fixed lead: unmet demand {int(t['lost'].sum())} units, of which {lost_zero} ({100 * lost_zero / max(1, int(t['lost'].sum())):.1f}%) in cycles whose order-up-to level was 0; "
      f"{int(zS.sum())} of {zS.size} series-cycles had S = 0")
