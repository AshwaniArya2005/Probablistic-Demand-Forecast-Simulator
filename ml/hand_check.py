"""Hand recomputation of one series over ten review weeks (four warm-up, six counted) with code that shares nothing with ml/simulator.py: plain Python integers and lists, the inputs recomputed from raw
sales where possible. It recomputes, for the quantile policy at alpha 0.90 and the naive rule at c = 2, the order-up-to levels, the whole day-by-day stock path, the orders, and each counted cycle's
demand, lost units and stock-days, and compares every quantity with the engine. The day-by-day table (which contains one series' daily sales) is written to the git-ignored data/processed and never
committed; only the pass/fail summary is written to docs/results/phase10_handcheck.md.
    uv run python ml/hand_check.py tuning        (F4 development fold; tuning data)
    uv run python ml/hand_check.py test          (test window; needs the touch-log row "Phase 10 diagnostic handcheck")"""
import math
import sys
from datetime import date

import numpy as np
import pandas as pd

import sim_run as sr
import simulator as sim
from config import PROCESSED, ROOT
from folds import touch_logged

P, L, R = 10, 3, 7
kind = sys.argv[1] if len(sys.argv) > 1 else "tuning"
if kind == "test" and not touch_logged("Phase 10 diagnostic handcheck"):
    raise SystemExit("refusing to run: no 'Phase 10 diagnostic handcheck' row in the touch log (docs/design.md section 14)")
run = (sr.dev_runs()[1] if kind == "tuning" else sr.test_runs()[0])
ids = sorted(run.tab.id.unique())
dem, _, item_of, seg_of = sr.load_market(ids)
days = dem.index
rv_dates = list(run.dates)
rv = [days.get_loc(d) for d in rv_dates]
K, W = len(rv), run.n_warmup
levels, _ = sr.levels_for(run, (2.0,), ids)
tab = run.tab.set_index(["id", "date"])
panel = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "date", "sales"]).set_index(["id", "date"]).sales


# ---- choose the series by a fixed rule: the first mid-velocity id (alphabetical) with at least one lost unit in the counted weeks under the quantile policy at alpha 0.90 ----
def engine(j):
    rep = sim.replay(dem.to_numpy(), np.array(rv), levels[("quantile", 0.90)], L)
    return sim.cycle_table(rep, dem.to_numpy(), np.array(rv), L, np.arange(W, K))


tq = engine(0)
mid = [j for j, s in enumerate(seg_of) if s == "mid"]
pick = next(j for j in mid if tq["lost"][:, j].sum() > 0)
sid = ids[pick]
col = [int(x) for x in dem[sid].to_numpy()]                                    # daily sales as python ints
checks, log_lines = [], []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# ---- inputs recomputed from raw sales ----
mu28_hand = [sum(col[d - 27:d + 1]) / 28.0 for d in rv]
mu28_tab = [float(tab.loc[(sid, dt), "mu28"]) for dt in rv_dates]
check("mu28 equals the mean of the 28 sales days ending on each review date", all(abs(a - b) < 1e-9 for a, b in zip(mu28_hand, mu28_tab)), f"max abs diff {max(abs(a - b) for a, b in zip(mu28_hand, mu28_tab)):.2e}")
S_naive_hand = [math.ceil(round(2.0 * m * P, 9)) for m in mu28_hand]
S_quant_hand = [math.ceil(round(float(tab.loc[(sid, dt), "q90"]), 9)) for dt in rv_dates]
check("naive levels ceil(2 x mu28 x P) match the engine's input", S_naive_hand == [int(x) for x in levels[("naive", 2.0)][:, pick]])
check("quantile levels ceil(q90) match the engine's input", S_quant_hand == [int(x) for x in levels[("quantile", 0.90)][:, pick]])
check("panel sales equal the demand matrix used by the engine", [int(panel.loc[(sid, d)]) for d in days[rv[0] - 3:rv[0] + 5]] == col[rv[0] - 3:rv[0] + 5])


# ---- independent day-by-day replay ----
def hand_replay(S):
    on, pipe, orders, log = S[0], {}, [0] * K, []
    for d in range(rv[0] + 1, rv[-1] + L + R + 1):
        arr = pipe.pop(d, 0)
        on += arr
        served = min(on, col[d])
        lost = col[d] - served
        on -= served
        q = 0
        if d in rv[1:]:
            k = rv.index(d)
            q = max(S[k] - (on + sum(pipe.values())), 0)
            orders[k] = q
            if q:
                pipe[d + L + 1] = pipe.get(d + L + 1, 0) + q
        log.append((d, arr, col[d], served, lost, on, q))
    return log, orders


for name, S_hand, key in (("quantile alpha 0.90", S_quant_hand, ("quantile", 0.90)), ("naive c = 2", S_naive_hand, ("naive", 2.0))):
    log, orders = hand_replay(S_hand)
    rep = sim.replay(dem.to_numpy(), np.array(rv), levels[key], L)
    eng_on = [int(rep.onhand[d, pick]) for d, *_ in log]
    eng_lost = [int(rep.lost[d, pick]) for d, *_ in log]
    check(f"{name}: end-of-day on-hand path ({len(log)} days) equals the engine's", eng_on == [r[5] for r in log])
    check(f"{name}: lost units per day equal the engine's", eng_lost == [r[4] for r in log])
    check(f"{name}: orders placed at each review equal the engine's", orders == [int(x) for x in rep.orders[:, pick]])
    tab_e = sim.cycle_table(rep, dem.to_numpy(), np.array(rv), L, np.arange(W, K))
    for i, k in enumerate(range(W, K)):
        lo, hi = rv[k] + L + 1, rv[k] + L + R
        seg = [r for r in log if lo <= r[0] <= hi]
        hand = (sum(r[2] for r in seg), sum(r[4] for r in seg), sum(r[5] for r in seg), int(sum(r[4] for r in seg) > 0))
        eng = (int(tab_e["demanded"][i, pick]), int(tab_e["lost"][i, pick]), int(tab_e["onhand"][i, pick]), int(tab_e["stockout"][i, pick]))
        check(f"{name}: counted cycle {i + 1} (review {rv_dates[k].date() if kind == 'tuning' else 'week ' + str(i + 1)}): demanded, lost, stock-days, stockout flag", hand == eng, f"hand {hand} engine {eng}")
    if kind == "tuning" or True:
        log_lines.append(f"\n## {name} (series {sid})\n\n| day | arrivals | demand | served | lost | on hand | order placed |\n|---|---|---|---|---|---|---|")
        log_lines += [f"| {days[r[0]].date()} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in log]

ok = all(c[1] for c in checks)
(PROCESSED / f"handcheck_{kind}_daily.md").write_text(f"# Day-by-day hand recomputation ({kind}); contains one series' daily sales, never commit\n" + "\n".join(log_lines), encoding="utf-8")
summary = f"""# Hand recomputation of one series (Phase 10 check, {kind})

Run {date.today()}. One mid-velocity series (chosen by a fixed rule: the first mid-velocity series, alphabetically, with at least one lost unit in the six counted weeks under the quantile policy at alpha 0.90;
id withheld), ten review weeks (four warm-up, six counted), {kind} data. Every quantity below was recomputed with plain-Python integers and lists that share no code with `ml/simulator.py` and compared with the engine:
the 28-day means from raw sales, the order-up-to levels of the naive rule (c = 2) and of the quantile policy (alpha 0.90), the full day-by-day stock path, the orders, and each counted cycle's demand, lost units,
stock-days and stockout flag. The day-by-day table itself contains one series' daily sales and is kept out of the repository.

**Result: {'all ' + str(len(checks)) + ' checks agree' if ok else 'DISAGREEMENT, see below'}.**

| check | agrees |
|---|---|
""" + "\n".join(f"| {n} | {'yes' if o else '**NO** ' + d} |" for n, o, d in checks) + "\n"
(ROOT / "docs" / "results" / f"phase10_handcheck_{kind}.md").write_text(summary, encoding="utf-8")
print(summary)
