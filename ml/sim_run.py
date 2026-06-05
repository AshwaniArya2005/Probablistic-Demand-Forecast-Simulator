"""Phase 10 driver: replays the four policies over their parameter grids on one or more independent runs (a run = one contiguous replay: a development fold, or the
whole test window), pools the runs, applies the pre-registered naive c-grid widening rule, and writes the report. Rules: docs/design.md section 12 (Phase 10 pre-run
rules and Phase 10 definitions). Development (F2 and F4, P = 10, L = 3), needs data/processed/sim_dev_*.parquet from ml/sim_rows.py:
    uv run python ml/sim_run.py dev
Outputs: docs/results/phase10_simulator_dev_F2F4.md and .csv. The test-window primary run is a separate entry point added only after its touch-log entry."""
import subprocess
import sys
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

import policy
import sim_policies as sp
import simulator as sim
import versions
from config import HOLIDAY_END, PROCESSED, ROOT
from folds import touch_logged

P, L = 10, 3                                        # base case: lead time 3, R = 7, so P = 10
SEGS = ("low", "mid", "high")
ALPHAS = policy.SERVICE
C_START = tuple(float(c) for c in np.round(np.arange(1.0, 3.0001, 0.25), 2))
BOOT_B, BOOT_SEED = 10_000, 0
# the naive c-grid frozen from the F2 + F4 development run (widening rule of design.md section 12, Phase 10 pre-run rules item 3, applied there; the cap c = 10 was reached)
FROZEN_C_GRID = C_START + tuple(float(c) for c in np.round(np.arange(3.5, 10.0001, 0.5), 2))
POLICIES = ("posthoc", "point", "quantile")


@dataclass
class Run:
    name: str
    tab: pd.DataFrame          # forecast rows (id, date, segment, mu28, scale, yhat, q80..q99) at the warm-up and review dates
    cals: list                 # calibration table of each model unit
    unit_dates: list           # review dates served by each unit, in order
    dates: pd.DatetimeIndex    # all review dates, warm-up first
    n_warmup: int
    boundary: pd.Timestamp = None     # cycles starting on or before this day form the "peak" sub-period (None: no split)
    P: int = 10                       # protection interval = R + L
    L: int = 3                        # nominal lead time (the policies always plan with it)
    lead: np.ndarray = None           # optional (K, N) realised lead time of each order (random-lead sensitivity)
    exclude_zero: bool = False        # hindsight diagnostic: drop cells whose protection-interval demand was zero, from every policy alike


def load_market(ids):
    """demand (days x ids, int), price (days x ids), item and velocity segment of each id"""
    panel = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "item_id", "segment", "date", "sales", "sell_price"])
    dem = panel.pivot(index="date", columns="id", values="sales").reindex(columns=ids)
    price = panel.pivot(index="date", columns="id", values="sell_price").reindex(columns=ids)
    assert (dem.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), "the day index must be daily"
    assert dem.loc["2014-01-01":].notna().all().all(), "every series must have sales on every day of the simulated years (rows before a series' first record are zero and unused)"
    dem, price = dem.fillna(0), price
    meta = panel.drop_duplicates("id").set_index("id").loc[ids]
    return dem.astype("int64"), price, meta.item_id.to_numpy(), meta.segment.to_numpy()


def levels_for(run, c_grid, ids, alphas=ALPHAS):
    """{(policy, setting): S (K, N)}: each review date's levels come from the model unit serving it; also {unit: {segment: sigma form}}"""
    out, forms, seen = {}, {}, []
    for u, (cal, ud) in enumerate(zip(run.cals, run.unit_dates)):
        ud = pd.DatetimeIndex(ud)
        lv, f = sp.all_levels(run.tab[run.tab.date.isin(ud)], cal, run.P, c_grid, alphas=alphas, dates=ud, ids=ids)
        for k, v in lv.items():
            out.setdefault(k, []).append(v)
        forms[u] = f
        seen.extend(ud)
    assert list(pd.DatetimeIndex(seen)) == list(run.dates), "units must cover the review dates in order, once each"
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}, forms


def replay_run(run, levels, dem):
    L, days = run.L, dem.index
    rv = np.array([days.get_loc(d) for d in run.dates])
    metric_k = np.arange(run.n_warmup, len(rv))
    tables = {}
    for k, S in levels.items():
        rep = sim.replay(dem.to_numpy(), rv, S, L, lead=run.lead)
        tables[k] = sim.cycle_table(rep, dem.to_numpy(), rv, L, metric_k, P=run.P)
    return tables, (days[rv[metric_k[0]] + L + 1], days[rv[-1] + L + sim.REVIEW])


def peak_mask(run):
    """metric cycles in the peak sub-period: cycle start day (review + L + 1) on or before run.boundary; all cycles when there is no boundary"""
    starts = pd.DatetimeIndex(run.dates[run.n_warmup:]) + pd.Timedelta(days=run.L + 1)
    return np.ones(len(starts), bool) if run.boundary is None else np.asarray(starts <= run.boundary)


def pooled(all_tables, key, item_of_id, items, col_mask=None, period=None):
    """item sums pooled over runs; period 'peak' / 'rest' / None; col_mask restricts the series (velocity segment)"""
    tot = None
    for run, tables, _ in all_tables:
        pm = peak_mask(run)
        cm = None if period is None else (pm if period == "peak" else ~pm)
        if cm is not None and not cm.any():
            continue
        t, ids = tables[key], np.asarray(item_of_id)
        if col_mask is not None:
            t, ids = {k: v[:, col_mask] for k, v in t.items()}, ids[col_mask]
        s = sim.item_sums(t, ids, items, cm, (t['zero'] == 0) if run.exclude_zero else None)
        tot = s if tot is None else {k: tot[k] + s[k] for k in s}
    return tot


def fmt4(x):
    return "n/a" if x != x else f"{x:.4f}"


def fill_of(s):
    return 1 - s["lost"].sum() / s["demanded"].sum()


def widen_naive(runs, ids, dem, all_tables, item_of_id, items):
    """the pre-registered c-grid widening rule (design.md section 12, Phase 10 pre-run rules, item 3): upward in steps of 0.5 (cap 10) until the naive pooled fill rate reaches
    the highest pooled fill rate any other policy has at alpha = 0.99, downward in steps of 0.25 (floor 0.25) until it is at or below the lowest at alpha = 0.80.
    all_tables is extended in place; returns the final grid and the log of steps."""
    other = lambda a, f: f(fill_of(pooled(all_tables, (p, a), item_of_id, items)) for p in POLICIES)
    top, bottom = other(0.99, max), other(0.80, min)
    grid, log = list(C_START), []
    fills = lambda: {c: fill_of(pooled(all_tables, ("naive", c), item_of_id, items)) for c in grid}

    def add(c):
        for run, tables, _ in all_tables:
            lv, _ = levels_for(run, (c,), ids, alphas=())
            new, _ = replay_run(run, lv, dem)
            tables.update(new)
        grid.append(c)

    while max(fills().values()) < top and max(grid) < 10:
        c = round(max(grid) + 0.5, 2)
        add(c)
        log.append(f"up: c = {c:.2f} added (max naive fill {max(fills().values()):.4f} vs top {top:.4f})")
    while min(fills().values()) > bottom and min(grid) > 0.25:
        c = round(max(min(grid) - 0.25, 0.25), 2)
        add(c)
        log.append(f"down: c = {c:.2f} added (min naive fill {min(fills().values()):.4f} vs bottom {bottom:.4f})")
    return sorted(grid), log, top, bottom


def curves(sums_of, keys, W):
    """(fill, cycle service, inventory) each (B, len(keys)) over the weights W for the ordered keys"""
    st = [sim.curve_stats(W, sums_of[k]) for k in keys]
    return tuple(np.stack([s[m] for s in st], axis=1) for m in ("fill", "cycle_service", "inv"))


def reductions(sums_of, comparator_keys, W, anchor="q"):
    """inventory reduction of the quantile policy vs a comparator curve at matched fill rate: (B, 4) at the quantile points (anchor 'q', pre-registered) or (B, m) at the
    comparator's own points (anchor 'c'); NaN where the other curve does not reach (no extrapolation)"""
    qf, _, qi = curves(sums_of, [("quantile", a) for a in ALPHAS], W)
    cf, _, cinv = curves(sums_of, comparator_keys, W)
    return sim.matched_reduction(qf, qi, cf, cinv, anchor)


def ci(x):
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def pct(x):
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}%"


def compare_table(all_tables, item_of_id, items, comparators, W, col_mask=None, period=None, anchor="q"):
    """rows: comparator; columns: reduction per setting (point estimate), mean over matched settings with its 95% item-bootstrap interval, share of resamples with a match.
    anchor 'q': settings are the quantile policy's alphas (pre-registered statistic); anchor 'c': the comparator's own settings (added statistic)."""
    keysets = {k: [(p, x) for x in ks] for k, (p, ks) in comparators.items()}
    sums_of = {}
    for ks in keysets.values():
        for key in ks:
            sums_of[key] = pooled(all_tables, key, item_of_id, items, col_mask, period)
    for a in ALPHAS:
        sums_of[("quantile", a)] = pooled(all_tables, ("quantile", a), item_of_id, items, col_mask, period)
    W1 = np.ones((1, len(items)))
    rows = []
    for name, ks in keysets.items():
        pt = reductions(sums_of, ks, W1, anchor)[0]
        bt = reductions(sums_of, ks, W, anchor)
        with np.errstate(all="ignore"):
            mean_b = np.nanmean(bt, axis=1)
            mean_pt = np.nanmean(pt) if (~np.isnan(pt)).any() else np.nan
        lo, hi = ci(mean_b)
        per = {f"alpha {a}": pct(pt[j]) for j, a in enumerate(ALPHAS)} if len(pt) == len(ALPHAS) else {f"alpha {a}": "-" for a in ALPHAS}
        rows.append({"comparator": name, **per, "settings matched": f"{int((~np.isnan(pt)).sum())} of {len(pt)}", "mean over matched": pct(mean_pt),
                     "95% interval (items)": f"[{pct(lo)}, {pct(hi)}]", "resamples with a match": f"{100 * np.mean(~np.isnan(bt).all(1)):.0f}%"})
    return pd.DataFrame(rows)


def cost_totals(all_tables, price):
    """{(key, rho): total cost pooled over the runs}; price = item-store median sell_price over each run's evaluated days"""
    total = {}
    for run, tables, (d0, d1) in all_tables:
        pr = price.loc[d0:d1].median()
        pr = pr.fillna(price.median()).fillna(price.stack().median()).to_numpy()
        for key, t in tables.items():
            keep = (t["zero"] == 0) if run.exclude_zero else 1
            for rho in sim.RHOS:
                h, s = sim.cost_parts(pr, (t["onhand"] * keep).sum(0), (t["lost"] * keep).sum(0), rho)
                total[(key, rho)] = total.get((key, rho), 0.0) + float(h.sum() + s.sum())
    return total


def cost_table(all_tables, price, dem, key_lists):
    """minimum total cost over each policy's own settings, per rho (units: currency, pooled over the runs); a minimum at either end of the policy's grid is flagged"""
    total = cost_totals(all_tables, price)
    rows = []
    for name, keys in key_lists.items():
        r = {"policy": name}
        for rho in sim.RHOS:
            best = min(keys, key=lambda k: total[(k, rho)])
            edge = " (edge of its grid)" if best in (keys[0], keys[-1]) and len(keys) > 1 else ""
            r[f"rho {rho} (critical ratio {rho / (rho + 1):.2f})"] = f"{total[(best, rho)]:,.0f} at {best[1]}{edge}"
        rows.append(r)
    return pd.DataFrame(rows)


def predictions_section(all_tables, item_of_id, items, seg_of_id, comps, W):
    """the three predictions of design.md section 12 (Phase 10 definitions, item 2), evaluated mechanically on the pooled test-window run"""
    one = np.ones((1, len(items)))
    stat = lambda p, a, per=None: sim.curve_stats(one, pooled(all_tables, (p, a), item_of_id, items, None, per))
    rows = []
    for p, name in (("quantile", "quantile policy"), ("point", "point policy (B2 / B2-sqrt)")):
        for a in ALPHAS:
            gp, gr = stat(p, a, "peak")["cycle_service"][0] - a, stat(p, a, "rest")["cycle_service"][0] - a
            rows.append({"policy": name, "alpha": a, "gap in peak": f"{gp:+.3f}", "gap in rest": f"{gr:+.3f}", "rest below target": "yes" if gr < 0 else "no",
                         "rest gap smaller than peak gap": "yes" if gr < gp else "no"})
    a_tab = pd.DataFrame(rows)
    q = {a: stat("quantile", a) for a in ALPHAS}
    marg = lambda lo, hi: (q[hi]["inv"][0] - q[lo]["inv"][0]) / (100 * (q[hi]["fill"][0] - q[lo]["fill"][0]))
    m1, m2 = marg(0.90, 0.95), marg(0.95, 0.99)
    peak99 = stat("quantile", 0.99, "peak")["cycle_service"][0]
    keys = {k: [(p, x) for x in ks] for k, (p, ks) in comps.items()}
    b3a, b2 = "B3a (post-hoc, the fair benchmark)", "B2 / B2-sqrt (point policy, normal sigma)"
    seg = np.asarray(seg_of_id)
    e_rows = []
    for label, mask in (("pooled", None), ("high", seg == "high"), ("mid", seg == "mid"), ("low", seg == "low")):
        sums_of = {k: pooled(all_tables, k, item_of_id, items, mask) for ks in keys.values() for k in ks}
        sums_of.update({("quantile", a): pooled(all_tables, ("quantile", a), item_of_id, items, mask) for a in ALPHAS})
        for anchor, aname in (("c", "comparator (defined for both)"), ("q", "quantile (pre-registered)")):
            v = {}
            for name in (b3a, b2):
                r = reductions(sums_of, keys[name], one, anchor)[0]
                v[name] = float(np.nanmean(r)) if (~np.isnan(r)).any() else float("nan")
            smaller = "n/a" if np.isnan(v[b3a]) or np.isnan(v[b2]) else ("yes" if v[b3a] < v[b2] else "no")
            e_rows.append({"scope": label, "anchor": aname, "vs B3a": pct(v[b3a]), "vs B2 / B2-sqrt": pct(v[b2]), "smaller vs B3a": smaller})
    quant = a_tab[a_tab.policy == "quantile policy"]
    a_ok = bool((quant["rest below target"] == "yes").all() and (quant["rest gap smaller than peak gap"] == "yes").all())
    a_pt = int((a_tab[a_tab.policy != "quantile policy"]["rest gap smaller than peak gap"] == "yes").sum())
    c_ok = all(r["smaller vs B3a"] == "yes" for r in e_rows if r["anchor"].startswith("comparator") and r["scope"] in ("pooled", "high", "mid"))
    return f"""## 0. Pre-registered predictions (design.md section 12, Phase 10 definitions, item 2), evaluated mechanically

**(a) Rest-period under-service.** Cycle service minus target alpha, peak (cycles starting on or before 2016-01-03) vs rest:

{a_tab.to_markdown(index=False)}

Quantile policy: rest below target at every alpha and by more than in the peak: **{'CONFIRMED' if a_ok else 'NOT CONFIRMED'}**. Point policy (relative claim only): the rest gap is smaller than the peak gap at {a_pt} of 4 alphas.

**(b) 0.99 is inefficient for the quantile policy.** Marginal inventory per percentage point of fill rate: 0.90 to 0.95: {m1:.2f} units per series-day; 0.95 to 0.99: {m2:.2f}.
Cycle service at 0.99 in the peak: {peak99:.4f} (prediction: at or above 0.99). Verdict: **{'CONFIRMED' if (m2 > m1 and peak99 >= 0.99) else 'NOT CONFIRMED'}** (marginal cost larger: {'yes' if m2 > m1 else 'no'}; peak service at or above 0.99: {'yes' if peak99 >= 0.99 else 'no'}).

**(c) A smaller edge over B3a than over the point policy** (mean inventory reduction at matched fill rate over the matched settings; the comparator-anchored statistic is the one defined for both comparators):

{pd.DataFrame(e_rows).to_markdown(index=False)}

Verdict on the comparator-anchored statistic, pooled and in the high and mid segments: **{'CONFIRMED' if c_ok else 'NOT CONFIRMED'}**.
"""


def main_report(title, runs, path, all_tables, grid, log, top, bottom, forms, item_of_id, seg_of_id, items, dem, price, extra="", predictions=False):
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    W = sim.bootstrap_weights(len(items), BOOT_B, BOOT_SEED)
    n_units = sum(len(r.cals) for r in runs)
    comps = {"B3a (post-hoc, the fair benchmark)": ("posthoc", ALPHAS),
             "B2 / B2-sqrt (point policy, normal sigma)": ("point", ALPHAS),
             "Naive (c x mu28 x P), widened grid": ("naive", tuple(grid))}
    # ---- curves ----
    rows = []
    label = {"posthoc": sp.LABEL["posthoc"], "point": "B2 / B2-sqrt: point policy, normal sigma (form per segment below)", "quantile": sp.LABEL["quantile"]}
    for p in POLICIES:
        for a in ALPHAS:
            s = pooled(all_tables, (p, a), item_of_id, items)
            st = sim.curve_stats(np.ones((1, len(items))), s)
            rows.append({"policy": label[p], "setting": f"alpha {a}", "fill rate": f"{st['fill'][0]:.4f}", "cycle service": f"{st['cycle_service'][0]:.4f}", "avg on-hand (units per series-day)": f"{st['inv'][0]:.2f}"})
    for c in grid:
        s = pooled(all_tables, ("naive", c), item_of_id, items)
        st = sim.curve_stats(np.ones((1, len(items))), s)
        rows.append({"policy": sp.LABEL["naive"], "setting": f"c {c:.2f}", "fill rate": f"{st['fill'][0]:.4f}", "cycle service": "-", "avg on-hand (units per series-day)": f"{st['inv'][0]:.2f}"})
        rows[-1]["cycle service"] = f"{st['cycle_service'][0]:.4f}"
    curve_df = pd.DataFrame(rows)
    # ---- segments and periods ----
    seg_tabs = {s: compare_table(all_tables, item_of_id, items, comps, W, col_mask=(np.asarray(seg_of_id) == s), anchor="c") for s in SEGS}
    per_tabs = {}
    for per in ("peak", "rest"):
        if any(peak_mask(r).any() if per == "peak" else (~peak_mask(r)).any() for r in runs) and any(r.boundary is not None for r in runs):
            per_tabs[per] = compare_table(all_tables, item_of_id, items, comps, W, period=per, anchor="c")
    gaps = []
    for per in (None, "peak", "rest"):
        if per is not None and per not in per_tabs:
            continue
        for p in POLICIES:
            r = {"period": per or "all", "policy": {"posthoc": "B3a", "point": "B2 / B2-sqrt", "quantile": "Quantile"}[p]}
            for a in ALPHAS:
                s = pooled(all_tables, (p, a), item_of_id, items, None, per)
                r[f"alpha {a}"] = f"{sim.curve_stats(np.ones((1, len(items))), s)['cycle_service'][0] - a:+.3f}"
            gaps.append(r)
    forms_txt = []
    for s in SEGS:
        used = [f[s] for run_f in forms for f in run_f.values()]
        forms_txt.append({"segment": s, "B2 (constant CV)": sum(x.startswith("constant-CV") and "both" not in x for x in used),
                          "B2 (constant CV, check failed for both forms)": sum("both" in x for x in used), "B2-sqrt": sum(x == "sqrt-scale" for x in used), "model versions": len(used)})
    keylists = {"B3a": [("posthoc", a) for a in ALPHAS], "B2 / B2-sqrt": [("point", a) for a in ALPHAS], "Quantile": [("quantile", a) for a in ALPHAS],
                "Naive": [("naive", c) for c in grid]}
    md = f"""# {title}

Run {date.today()} at code commit `{head}`. L = {runs[0].L}, R = 7 (P = {runs[0].P}); {len(runs)} run(s), {n_units} model unit(s); {len(items)} items, {len(seg_of_id)} series.
Item cluster bootstrap: {BOOT_B:,} resamples over items, seed {BOOT_SEED}; it reflects which items were sampled, **not variation between periods**. The comparator curves have four points (the
alpha grid), so interpolation is coarse; a quantile point outside a comparator's fill-rate range is "not matched" (no extrapolation).
{extra}
**Order and labels (design.md, Phase 10 definitions, item 3):** B3a (point forecast + empirical residual quantiles, post-hoc) is the fair benchmark and comes first; B2 is the pre-specified
point policy (normal sigma, constant CV) and B2-sqrt its sqrt-scale variant where the Phase 9 check selects it; the naive rule comes last. The framing rule applies to every comparison.

{predictions_section(all_tables, item_of_id, items, seg_of_id, comps, W) if predictions else ''}
## 1. Inventory reduction of the quantile policy at matched fill rate (pooled)

Inventory reduction = 1 - inventory(quantile) / inventory(comparator at the same pooled fill rate).

**1a. Quantile-anchored (the pre-registered statistic):** at each of the quantile policy's four points, the comparator's inventory interpolated at that fill rate.

{compare_table(all_tables, item_of_id, items, comps, W).to_markdown(index=False)}

**1b. Comparator-anchored (added before the test-window run; design.md):** at each of the comparator's own settings, the quantile policy's inventory interpolated at that fill rate. Defined wherever the quantile curve reaches; the point policy is over-protective, so the quantile-anchored version has no match for it.

{compare_table(all_tables, item_of_id, items, comps, W, anchor="c").to_markdown(index=False)}

## 2. By velocity segment

Reading 1a and 1b together: with four points per curve the interpolation is coarse, and inventory is convex in fill rate, so a straight line between two points lies above the true curve.
Interpolating the comparator (1a) therefore tends to overstate the reduction and interpolating the quantile policy (1b) tends to understate it; **treat 1a and 1b as an upper and a lower bound** rather than as two estimates.

## 2. By velocity segment (comparator-anchored, defined for every comparator)

""" + "\n\n".join(f"**{s}**\n\n{seg_tabs[s].to_markdown(index=False)}" for s in SEGS) + f"""

## 3. Achieved cycle service minus target alpha, by period (calibration check)

{pd.DataFrame(gaps).to_markdown(index=False)}
""" + ("\n## 4. By period, inventory reduction at matched fill rate (comparator-anchored)\n\n" + "\n\n".join(f"**{k}**\n\n{v.to_markdown(index=False)}" for k, v in per_tabs.items()) + "\n" if per_tabs else "") + f"""
## 5. Policy curves (pooled over runs)

{curve_df.to_markdown(index=False)}

Naive c-grid widening (rule 3): start {list(C_START)}; top target (highest fill among the other policies at alpha 0.99) {fmt4(top)}; bottom target (lowest at alpha 0.80) {fmt4(bottom)}.
{chr(10).join('- ' + x for x in log) if log else '- no widening was needed'}
Final grid: {grid}

## 6. Sigma form used by the point policy (Phase 9 check, per model unit and segment)

{pd.DataFrame(forms_txt).to_markdown(index=False)}

## 7. Total cost: minimum over each policy's own settings, per stockout-to-holding ratio rho (currency units, pooled; a sensitivity, not the headline)

{cost_table(all_tables, price, dem, keylists).to_markdown(index=False)}
"""
    path.write_text(md, encoding="utf-8")
    curve_df.to_csv(path.with_suffix(".csv"), index=False)
    return md


def evaluate(runs, grid=None):
    """replay everything; the naive grid is widened by the pre-registered rule unless a frozen `grid` is given (the test-window run uses the frozen one)"""
    ids = sorted(runs[0].tab.id.unique())
    dem, price, item_of_id, seg_of_id = load_market(ids)
    items = sorted(set(item_of_id))
    all_tables, forms = [], []
    for run in runs:
        lv, f = levels_for(run, C_START if grid is None else grid, ids)
        tables, span = replay_run(run, lv, dem)
        all_tables.append((run, tables, span))
        forms.append(f)
    if grid is None:
        grid, log, top, bottom = widen_naive(runs, ids, dem, all_tables, item_of_id, items)
    else:
        grid, log, top, bottom = sorted(grid), ["the grid is frozen from the development run; it was not widened on this data"], float("nan"), float("nan")
    return dict(all_tables=all_tables, grid=grid, log=log, top=top, bottom=bottom, forms=forms, item_of_id=item_of_id, seg_of_id=seg_of_id, items=items, dem=dem, price=price)


def dev_runs():
    runs = []
    for fold in ("F2", "F4"):
        tab = pd.read_parquet(PROCESSED / f"sim_dev_{fold}_P{P}_tab.parquet")
        cal = pd.read_parquet(PROCESSED / f"sim_dev_{fold}_P{P}_cal.parquet")
        dates = pd.DatetimeIndex(sorted(tab.date.unique()))
        assert len(dates) == 16 and (dates.dayofweek == 6).all()
        boundary = HOLIDAY_END - pd.Timedelta(days=364) if fold == "F2" else None       # F2: the analogue of the holiday peak one year earlier (2015-01-04)
        runs.append(Run(fold, tab, [cal], [list(dates)], dates, 4, boundary))
    return runs


def test_runs(P=10, L=3, prefix="sim_test", leads=None, exclude_zero=False):
    """the test-window replay input: model versions v0..v4 serve their review dates (design.md section 5), four warm-up reviews then the 25 test reviews. leads: list of
    (seed, (K, N) lead times) for the random-lead sensitivity, one run per draw; exclude_zero: the hindsight diagnostic."""
    tabs, cals, uds = [], [], []
    for v in versions.CUTOFFS:
        tabs.append(pd.read_parquet(PROCESSED / f"{prefix}_v{v}_P{P}_tab.parquet"))
        cals.append(pd.read_parquet(PROCESSED / f"{prefix}_v{v}_P{P}_cal.parquet"))
        uds.append(list(versions.use_dates(v)))
    tab, dates = pd.concat(tabs, ignore_index=True), versions.review_dates()
    kw = dict(P=P, L=L, exclude_zero=exclude_zero)
    if leads is None:
        return [Run("test window", tab, cals, uds, dates, 4, HOLIDAY_END, **kw)]
    return [Run(f"test window, lead draw {s}", tab, cals, uds, dates, 4, HOLIDAY_END, lead=lead, **kw) for s, lead in leads]


def primary_runs():
    return test_runs()


def lead_draws(K=29, N=300, n=10):
    """n independent draws of the lead time of every order, Uniform{2, 3, 4} (mean 3, the nominal), seeds 0 .. n - 1"""
    return [(s, np.random.default_rng(s).integers(2, 5, size=(K, N))) for s in range(n)]


def summary(ev):
    """pooled fill rate, cycle service and inventory per policy setting, and the comparator-anchored mean reduction against B3a and B2"""
    one = np.ones((1, len(ev["items"])))
    args = (ev["item_of_id"], ev["items"])
    out = {}
    for key in ev["all_tables"][0][1]:
        st = sim.curve_stats(one, pooled(ev["all_tables"], key, *args))
        out[key] = dict(fill=float(st["fill"][0]), cs=float(st["cycle_service"][0]), inv=float(st["inv"][0]))
    sums_of = {k: pooled(ev["all_tables"], k, *args) for p in POLICIES for k in [(p, a) for a in ALPHAS]}
    for name in ("posthoc", "point"):
        r = reductions(sums_of, [(name, a) for a in ALPHAS], one, "c")[0]
        out[f"red_{name}"] = float(np.nanmean(r)) if (~np.isnan(r)).any() else float("nan")
    return out


def compare_with_base(base, sens):
    rows = []
    for p, name in (("posthoc", "B3a"), ("point", "B2 / B2-sqrt"), ("quantile", "Quantile")):
        for a in ALPHAS:
            b, s = base[(p, a)], sens[(p, a)]
            rows.append({"policy": name, "alpha": a, "fill base": f"{b['fill']:.4f}", "fill here": f"{s['fill']:.4f}", "cycle service base": f"{b['cs']:.4f}", "cycle service here": f"{s['cs']:.4f}",
                         "avg on-hand base": f"{b['inv']:.2f}", "avg on-hand here": f"{s['inv']:.2f}"})
    return pd.DataFrame(rows)


def expected_outcomes(kind, base, sens):
    """the expected outcomes committed in design.md before the run, as (statement, holds) pairs"""
    q = lambda d, k, a: d[("quantile", a)][k]
    out = []
    if kind == "L7":
        out.append(("quantile policy: cycle service above target at every alpha", all(q(sens, "cs", a) > a for a in ALPHAS)))
        out.append(("comparator-anchored mean reduction against B3a is positive and within 8 percentage points of the base case", sens["red_posthoc"] > 0 and abs(sens["red_posthoc"] - base["red_posthoc"]) <= 0.08))
        out.append(("B2 still over-protects: its cycle service exceeds target by more than 0.05 at alpha 0.80", sens[("point", 0.80)]["cs"] - 0.80 > 0.05))
        out.append(("quantile policy holds more inventory than in the base case at every alpha", all(q(sens, "inv", a) > q(base, "inv", a) for a in ALPHAS)))
    elif kind == "randlead":
        out.append(("quantile policy: cycle service below the base case at every alpha", all(q(sens, "cs", a) < q(base, "cs", a) for a in ALPHAS)))
        out.append(("quantile policy: fill rate below the base case at every alpha", all(q(sens, "fill", a) < q(base, "fill", a) for a in ALPHAS)))
        out.append(("comparator-anchored mean reduction against B3a is positive", sens["red_posthoc"] > 0))
    elif kind == "nofp":
        out.append(("quantile policy: cycle service within 0.01 of the base case at every alpha", all(abs(q(sens, "cs", a) - q(base, "cs", a)) <= 0.01 for a in ALPHAS)))
        out.append(("quantile policy: average on-hand within 5% of the base case at every alpha", all(abs(q(sens, "inv", a) / q(base, "inv", a) - 1) <= 0.05 for a in ALPHAS)))
        out.append(("comparator-anchored mean reduction against B3a within 5 percentage points of the base case", abs(sens["red_posthoc"] - base["red_posthoc"]) <= 0.05))
    elif kind == "zerodemand":
        allkeys = [k for k in sens if isinstance(k, tuple)]
        out.append(("fill rate identical to the base case for every policy setting (dropped cells had no demand)", all(abs(sens[k]["fill"] - base[k]["fill"]) < 1e-9 for k in allkeys)))
        out.append(("cycle service below the base case for every quantile, B3a and B2 setting", all(sens[(p, a)]["cs"] < base[(p, a)]["cs"] for p in POLICIES for a in ALPHAS)))
        out.append(("comparator-anchored mean reduction against B3a is positive", sens["red_posthoc"] > 0))
    return out


def sens_report(kind, title, out, extra_note, grid=FROZEN_C_GRID):
    base = evaluate(test_runs(), FROZEN_C_GRID)
    runs = {"L7": lambda: test_runs(P=14, L=7), "randlead": lambda: test_runs(leads=lead_draws()), "nofp": lambda: test_runs(prefix="sim_test_nofp"),
            "zerodemand": lambda: test_runs(exclude_zero=True), "costext": lambda: test_runs()}[kind]()
    ev = evaluate(runs, grid)
    b, s = summary(base), summary(ev)
    exp = expected_outcomes(kind, b, s)
    sec = ""
    if exp:
        sec = "## 0. Expected outcomes (design.md, Phase 10 sensitivities entry), evaluated mechanically\n\n" + "\n".join(
            f"- {'**HELD**' if ok else '**DID NOT HOLD**'}: {txt}" for txt, ok in exp) + "\n\n" + compare_with_base(b, s).to_markdown(index=False) + \
            f"\n\nComparator-anchored mean reduction of the quantile policy against B3a: base {100 * b['red_posthoc']:.1f}%, here {100 * s['red_posthoc']:.1f}%; against B2 / B2-sqrt: base {100 * b['red_point']:.1f}%, here {100 * s['red_point']:.1f}%.\n"
    forms = [ev["forms"][0]] if kind == "randlead" else ev["forms"]
    md = main_report(title, runs, out, ev["all_tables"], ev["grid"], ev["log"], ev["top"], ev["bottom"], forms, ev["item_of_id"], ev["seg_of_id"], ev["items"], ev["dem"], ev["price"],
                     extra_note + "\n\n" + sec)
    return md, ev, base


def costext_report(out):
    """post-hoc: the primary run's cost sweep with the naive grid extended below its lower edge (design.md, Phase 10 sensitivities entry). Nothing else changes."""
    grid = tuple(sorted(set(FROZEN_C_GRID) | {0.25, 0.5, 0.75}))
    ev = evaluate(test_runs(), grid)
    prim = pd.read_csv(ROOT / "docs" / "results" / "phase10_simulator_primary.csv")
    got = pooled(ev["all_tables"], ("quantile", 0.8), ev["item_of_id"], ev["items"])
    assert abs(fill_of(got) - float(prim[(prim.iloc[:, 0].str.startswith("Quantile")) & (prim.iloc[:, 1] == "alpha 0.8")].iloc[0, 2])) < 1e-4, "the base-case numbers must reproduce the primary run"
    total = cost_totals(ev["all_tables"], ev["price"])
    keylists = {"B3a": [("posthoc", a) for a in ALPHAS], "B2 / B2-sqrt": [("point", a) for a in ALPHAS], "Quantile": [("quantile", a) for a in ALPHAS], "Naive, primary grid": [("naive", c) for c in FROZEN_C_GRID],
                "Naive, extended grid (post-hoc)": [("naive", c) for c in grid]}
    curve = pd.DataFrame([{"c": c, **{f"rho {r}": f"{total[(('naive', c), r)]:,.0f}" for r in sim.RHOS}} for c in grid if c <= 3.0])
    md = f"""# Phase 10 cost sweep with the naive grid extended (post-hoc)

Run {date.today()} at code commit `{subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}`. **Post-hoc**: made after the primary run showed the naive rule's rho 4 minimum at
the lower edge of its grid (c = 1.00). The naive grid is extended downward by 0.25 (floor 0.25, the pre-registered floor); nothing else changes and the base-case numbers reproduce the primary run.
Test-window data, logged in the touch log as its own entry.

## Minimum total cost per policy, per rho (currency units; a minimum at either end of a grid is flagged)

{cost_table(ev["all_tables"], ev["price"], ev["dem"], keylists).to_markdown(index=False)}

## Naive rule: total cost by c (c up to 3.00)

{curve.to_markdown(index=False)}

The target-driven policies' minima at alpha 0.80 cannot be extended (no quantile models between 0.50 and 0.80), so those results stay **unresolved**.
"""
    out.write_text(md, encoding="utf-8")
    return md


SENS = {"L7": ("Phase 10 sensitivity L7", "phase10_sens_L7.md", "Phase 10 sensitivity: lead time 7 (protection interval 14)",
               "**Sensitivity: lead time 7** (P = 14, cycles t + 8 .. t + 14); model versions refit at horizon 14 by `ml/sim_rows.py testP14`; everything else as the primary run."),
        "randlead": ("Phase 10 sensitivity randlead", "phase10_sens_randlead.md", "Phase 10 sensitivity: random lead time Uniform{2, 3, 4}",
                     "**Sensitivity: random lead time.** Each order's lead time is drawn Uniform{2, 3, 4} (mean 3); all policies plan with the nominal 3; 10 independent draws (seeds 0 to 9) are pooled; "
                     "cycles keep their nominal windows. With R = 7 and leads of 2 to 4 days no order can overtake an earlier one, so no crossing occurs in this design (the engine supports it and is tested)."),
        "nofp": ("Phase 10 sensitivity nofp", "phase10_sens_nofp.md", "Phase 10 sensitivity: no future-price inputs",
                 "**Sensitivity: no future-price inputs.** Quantile and mean models refit without the planned-window price features (the Phase 7 `nofutprice` arm) by `ml/sim_rows.py testnofp`; B2 and B3a use the same refit point forecast."),
        "zerodemand": ("Phase 10 diagnostic zerodemand", "phase10_diag_zerodemand.md", "Phase 10 diagnostic: excluding zero-demand cycles (hindsight)",
                       "**Hindsight diagnostic, never the headline.** Every (cycle, series) cell whose realised demand over the protection interval t + 1 .. t + P was zero is dropped from every policy's metrics alike."),
        "costext": ("Phase 10 cost extension", "phase10_cost_extended.md", "", "")}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if what == "primary":
        out = ROOT / "docs" / "results" / "phase10_simulator_primary.md"
        if not touch_logged():
            raise SystemExit("refusing to run: no 'Phase 10 primary run' row in the touch log (docs/design.md section 14)")
        if out.exists():
            raise SystemExit(f"{out.name} exists: the primary run is made once (design.md, Phase 10 pre-run rules item 2); a re-run needs a logged reason and a new touch-log entry")
        runs = primary_runs()
        ev = evaluate(runs, FROZEN_C_GRID)
        extra = ("**The primary test-window run** (design.md, Phase 10 pre-run rules item 2): base case L = 3, R = 7; four warm-up reviews (25 Oct to 15 Nov) excluded, 25 review dates counted "
                 "(22 Nov to 8 May); model versions v0 to v4 serve their review dates. Peak = cycles starting on or before 2016-01-03 (the holiday peak), rest = later cycles.")
        print(main_report("Phase 10 simulator, primary test-window run", runs, out, ev["all_tables"], ev["grid"], ev["log"], ev["top"], ev["bottom"], ev["forms"], ev["item_of_id"],
                          ev["seg_of_id"], ev["items"], ev["dem"], ev["price"], extra, predictions=True))
        raise SystemExit(0)
    if what in SENS:
        key, fname, title, note = SENS[what]
        out = ROOT / "docs" / "results" / fname
        if not touch_logged(key):
            raise SystemExit(f"refusing to run: no '{key}' row in the touch log (docs/design.md section 14)")
        if out.exists():
            raise SystemExit(f"{out.name} exists: each test-window run is made once; a re-run needs a logged reason and a new touch-log entry")
        if what == "costext":
            print(costext_report(out))
        else:
            print(sens_report(what, title, out, note)[0])
        raise SystemExit(0)
    if what != "dev":
        raise SystemExit("usage: sim_run.py dev | primary | " + " | ".join(SENS))
    runs = dev_runs()
    for r in runs:
        from folds import assert_tuning_only
        assert_tuning_only(r.dates, P)
    ev = evaluate(runs)
    extra = ("Development on **F2 (holiday fold) and F4 (summer fold) only**, tuning data (design.md, Phase 10 definitions, item 1); warm-up = the four Sundays before each fold's first origin. "
             "For F2 the peak sub-period is the analogue of the holiday peak one year earlier (cycles starting on or before 2015-01-04); F4 counts as rest. "
             "**Not the primary result and not a test of the pre-registered predictions** (those are evaluated on the test window only).")
    md = main_report("Phase 10 simulator, development run on F2 and F4", runs, ROOT / "docs" / "results" / "phase10_simulator_dev_F2F4.md", ev["all_tables"], ev["grid"], ev["log"],
                     ev["top"], ev["bottom"], ev["forms"], ev["item_of_id"], ev["seg_of_id"], ev["items"], ev["dem"], ev["price"], extra)
    print(md)
