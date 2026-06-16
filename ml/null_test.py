"""Shuffled-demand null test (default: tuning folds F2 and F4, no test-window data; `null_test.py test` repeats it on the test window and needs the touch-log row "Phase 11 null test test window"). If the simulator or the policy levels leaked information about realised demand, or if a reported edge were an
artefact of the matching statistic, the quantile policy's service and its edge over B3a would survive when realised demand is made independent of the forecasts. Two nulls, 20 seeds each, the
forecasts untouched: (1) BLOCK: each series' realised demand is permuted in whole-week blocks within the fold's simulated span (levels and weekday pattern kept, timing destroyed);
(2) SWAP: each series receives another series' realised demand (nothing left of the link between a series' forecasts and its demand). The expected outcomes are in docs/design.md (Phase 11 decisions
entry), written before this ran. Run: uv run python ml/null_test.py"""
import sys
from datetime import date

import numpy as np
import pandas as pd

import sim_run as sr
import simulator as sim
from config import ROOT
from folds import touch_logged

N_PERM = 20


def block_permute(seed, spans):
    """permute each series' demand in whole-week blocks inside each (start, end) day span (positions), independently per series"""
    def f(dem):
        rng = np.random.default_rng(seed)
        a = dem.to_numpy().copy()
        for s, e in spans:
            n = (e - s + 1) // 7
            for j in range(a.shape[1]):
                blocks = a[s:s + 7 * n, j].reshape(n, 7)
                a[s:s + 7 * n, j] = blocks[rng.permutation(n)].reshape(-1)
        return pd.DataFrame(a, index=dem.index, columns=dem.columns)
    return f


def swap(seed):
    def f(dem):
        rng = np.random.default_rng(seed)
        n = dem.shape[1]
        while True:
            perm = rng.permutation(n)
            if (perm != np.arange(n)).all():
                break
        return pd.DataFrame(dem.to_numpy()[:, perm], index=dem.index, columns=dem.columns)
    return f


def summarise(ev):
    s = sr.summary(ev)
    return {"cs90": s[("quantile", 0.90)]["cs"], "cs95": s[("quantile", 0.95)]["cs"], "fill90": s[("quantile", 0.90)]["fill"], "red_b3a": s["red_posthoc"], "red_b2": s["red_point"]}


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    if kind == "test" and not touch_logged("Phase 11 null test test window"):
        raise SystemExit("refusing to run: no 'Phase 11 null test test window' row in the touch log (docs/design.md section 14)")
    runs = sr.dev_runs() if kind == "tuning" else sr.test_runs()
    days = sr.load_market(sorted(runs[0].tab.id.unique()))[0].index
    spans = [(days.get_loc(r.dates[0]) + 1, days.get_loc(r.dates[-1]) + r.L + 7) for r in runs]
    real = summarise(sr.evaluate(runs, sr.FROZEN_C_GRID))
    out = {"real demand": [real]}
    for name, make in (("BLOCK (weekly blocks permuted within series)", lambda s: block_permute(s, spans)), ("SWAP (each series gets another series' demand)", swap)):
        out[name] = [summarise(sr.evaluate(runs, sr.FROZEN_C_GRID, demand_transform=make(s))) for s in range(N_PERM)]
    rows = []
    for name, lst in out.items():
        d = pd.DataFrame(lst)
        rows.append({"demand": name, "draws": len(d), **{f"{k} mean": f"{d[k].mean():.4f}" for k in d.columns},
                     "red_b3a 2.5%-97.5% over draws": "-" if len(d) == 1 else f"[{100 * d.red_b3a.quantile(.025):.1f}%, {100 * d.red_b3a.quantile(.975):.1f}%]",
                     "draws with red_b3a below the real value": "-" if len(d) == 1 else f"{int((d.red_b3a < real['red_b3a']).sum())} of {len(d)}"})
    md = f"""# Shuffled-demand null test ({"tuning folds F2 and F4" if kind == "tuning" else "test window"}, report-only)

Run {date.today()}. Base case L = 3 (P = 10), 300 series. `cs90`, `cs95`, `fill90`: the quantile policy's cycle service at alpha 0.90 and 0.95 and its fill rate at 0.90; `red_b3a`, `red_b2`: comparator-anchored mean
inventory reduction of the quantile policy against B3a and B2 / B2-sqrt (means over the settings that matched; NaN draws are dropped from the means). Forecasts are untouched, only the realised demand is replaced.

{pd.DataFrame(rows).to_markdown(index=False)}
"""
    (ROOT / "docs" / "results" / f"phase11_null_test_{kind}.md").write_text(md, encoding="utf-8")
    print(md)
