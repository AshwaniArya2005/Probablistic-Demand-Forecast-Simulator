"""Phase 7: quantile XGBoost on the tuning folds. Every rule is pre-registered in docs/design.md section 12 (Phase 7 pre-registration and its
amendment) and committed before the first score. One model version per fold and horizon: fit set ends at c_F - 84 days, the 12-week calibration
window (Sunday origins) is used only for sigma and conformal offsets, evaluation on the fold's 12 Sunday origins.
Stages (resumable, cached in data/processed/phase7_cache.json): A floor -> B search -> C decision -> D feature groups + sensitivities -> E final,
benchmarks and conformal -> report.  Run: uv run python ml/tune_quantile.py [floor|search|groups|final|report|all]"""
import functools
import json
import os
import subprocess
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

import conformal
import metrics
import quantiles
import selection
import versions
from config import PROCESSED, ROOT
from features import PS, STATIC, columns
from folds import FOLDS, assert_tuning_only, origins, select
from learned import XGBQ
from models import REGISTRY

SHARD, NSHARD = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)          # split the fits over processes: tune_quantile.py <stage> <i> <n>
CACHE = PROCESSED / ("phase7_cache.json" if NSHARD == 1 else f"phase7_cache_{SHARD}.json")      # each process writes its own file, all are merged on start
NTHREAD = int(os.environ.get("XGB_THREADS", 5))                                                    # one fixed thread count for every XGBoost fit
CELLS = [(f, P) for f in FOLDS for P in PS]
ALPHAS = XGBQ.ALPHAS
SERVICE = (0.80, 0.90, 0.95, 0.99)
FLOORS = (1 / 28, 1 / 14, 1 / 7)
FLOOR_CFG = dict(max_depth=3, min_child_weight=30)
GRID = [dict(max_depth=d, min_child_weight=w) for d in (3, 6) for w in (30, 5)]          # simplest first
POINT_CFG = dict(max_depth=3, min_child_weight=30)                                        # Phase 6 selected XGBoost config (both variants)
RF_CFG = dict(max_depth=10, min_samples_leaf=20)                                          # Phase 6 selected RF config (raw), comparison only
SEGS = ("low", "mid", "high")

feats = pd.read_parquet(PROCESSED / "features.parquet")
seg = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "segment"]).drop_duplicates("id").set_index("id").segment
EV = sorted({c[3:-3] for c in feats.columns if c.startswith("ev_") and c.endswith("_p7")})
cache = {(e["k"], tuple(e["c"])): e["r"] for f in sorted(PROCESSED.glob("phase7_cache*.json")) for e in json.loads(f.read_text())}


mine = {(e["k"], tuple(e["c"])): e["r"] for e in json.loads(CACHE.read_text())} if CACHE.exists() else {}      # this process's own results


def put(k, c, r):
    cache[(k, c)] = mine[(k, c)] = r


def save_cache():
    CACHE.write_text(json.dumps([{"k": k, "c": list(c), "r": r} for (k, c), r in mine.items()], default=float))


@functools.lru_cache(maxsize=1)          # one cell at a time keeps each shard near 1 GB
def cell_data(fold, P):
    c = origins(fold).min()
    fit = feats[versions.fit_mask(feats.date, c, P)]
    sel = select(feats, fold, P).reset_index(drop=True)
    cal = feats[versions.calibration_sundays(feats.date, c, P)].reset_index(drop=True)
    for d in (fit, cal):
        assert_tuning_only(d.date, P)
    return fit, sel, cal, metrics.naive_scale(feats, P, before=c)


def cols_for(P, variant, arm):
    """base set = the variant's Phase 6 recorded set (no ids; year-ago only for the normalised variant); arms are '+'-joined additions/sensitivities"""
    cols = [c for c in columns(P) if c not in STATIC]
    if variant == "raw":
        cols = [c for c in cols if c != f"sum_364_{P}"]
    for a in arm.split("+"):
        if a == "ev":
            cols += [f"ev_{n}_p{P}" for n in EV]
        elif a == "xs":
            cols += ["other_zero_run_91", "other_zero_28"]
        elif a == "ids_all":
            cols += list(STATIC)
        elif a == "ids_noitem":
            cols += [c for c in STATIC if c != "item_id"]
        elif a == "yearago_flip":
            cols = cols + [f"sum_364_{P}"] if variant == "raw" else [c for c in cols if c != f"sum_364_{P}"]
        elif a == "nofutprice":
            cols = [c for c in cols if not c.startswith(("price_mean_rel", "price_min_rel"))]
    return cols


def qrecord(Q, rows, P, scale):
    """all reported quantile metrics for a (n, 6) forecast on the evaluation rows"""
    y, ids = rows[f"y_p{P}"].to_numpy("float64"), rows.id
    sg = ids.map(seg).to_numpy()
    ps = pd.concat([metrics.scaled_pinball_by_series(y, Q[:, j], a, ids, scale) for j, a in enumerate(ALPHAS)], axis=1)
    per = ps.mean(axis=1)
    r = dict(sp_mean=per.mean(), sp_median=per.median(), sp_excluded=int(per.isna().sum()), wape_median=metrics.wape(y, Q[:, 1]),
             interval80=metrics.interval_coverage_discrete(y, Q[:, 0], Q[:, 3]))
    m, _, _ = metrics.mase(y, Q[:, 1], ids, scale)
    r["mase_median_fc"] = m
    for j, a in enumerate(ALPHAS):
        r[f"sp_{a}"] = ps.iloc[:, j].mean()
    for s in SEGS:
        r[f"sp_{s}"] = per[per.index.map(seg) == s].mean()
    for a in SERVICE:
        j = ALPHAS.index(a)
        r.update({f"{k}_{a}": v for k, v in metrics.coverage_discrete(y, Q[:, j], a).items()})
        for s in SEGS:
            c = metrics.coverage_discrete(y[sg == s], Q[sg == s, j], a)
            r[f"cov_hi_{a}_{s}"], r[f"cov_lo_{a}_{s}"] = c["cov_hi"], c["cov_lo"]
    return r


def scale_of(rows, P, floor):
    return pd.Series(np.fmax(rows.mean_28.to_numpy("float64"), floor) * P, index=rows.index)


def key(s):
    return json.dumps(s, sort_keys=True)


def run(specs, label):
    """specs: (variant, cfg, arm, floor); fits every uncached (spec, cell), cell-outermost so cell data is built once."""
    pending = [(s, c) for c in CELLS for s in specs if (key(s), c) not in cache]
    todo = pending[SHARD::NSHARD]
    print(f"[{label}] shard {SHARD}/{NSHARD}: {len(todo)} of {len(pending)} pending fits ({len(specs) * len(CELLS) - len(pending)} already cached)", flush=True)
    t0, n = time.time(), 0
    for (var, cfg, arm, floor), (fold, P) in todo:
        fit, sel, cal, nscale = cell_data(fold, P)
        t = time.time()
        m = XGBQ(P, seed=0, columns=cols_for(P, var, arm), normalize=(var == "norm"), floor=floor or 1 / 14, nthread=NTHREAD, **cfg).fit(fit, fit[f"y_p{P}"])
        secs = time.time() - t
        r = qrecord(m.predict_quantiles(sel, ALPHAS), sel, P, nscale)
        r.update(fold=fold, P=P, n_fit=len(fit), fit_seconds=round(secs, 1), n_rounds=int(m.n_rounds_), hit_cap=bool(m.hit_cap_),
                 crossing_share=m.crossing_share(sel))
        put(key((var, cfg, arm, floor)), (fold, P), r)
        n += 1
        if n % 4 == 0 or n == len(todo):
            save_cache()
            print(f"  {n}/{len(todo)} fits, {time.time() - t0:.0f}s elapsed (last fit {secs:.0f}s)", flush=True)


def cells_of(s):
    return pd.DataFrame([cache[(key(s), c)] for c in CELLS])


def mean_sp(s):
    return cells_of(s).sp_mean.mean()


def mean_spmed(s):
    return cells_of(s).sp_median.mean()


def spec(var, cfg, arm="base", floor=None):
    return (var, cfg, arm, floor if var == "norm" else None)


def summarise(fn):
    """print a summary only when every shard's results are merged in; a shard that finishes first has an incomplete view (its cache is read at start)"""
    try:
        fn()
    except KeyError:
        print("(summary skipped: other shards' results are not merged into this process; see the report)", flush=True)


def stage_floor():
    run([spec("norm", FLOOR_CFG, floor=f) for f in FLOORS], "A floor")

    def show():
        v = {f: mean_sp(spec("norm", FLOOR_CFG, floor=f)) for f in FLOORS}
        print("floor candidates (mean scaled pinball):", {round(k, 4): round(x, 4) for k, x in v.items()}, "->", round(selection.choose_floor_rel(v), 4), flush=True)
    summarise(show)


def chosen_floor():
    return selection.choose_floor_rel({f: mean_sp(spec("norm", FLOOR_CFG, floor=f)) for f in FLOORS})


def selected_cfg(var, floor):
    return selection.simplest_within_rel(GRID, [mean_sp(spec(var, c, floor=floor)) for c in GRID])


def stage_search(floor):
    run([spec(v, c, floor=floor) for v in ("raw", "norm") for c in GRID], "B search")


def adopted(floor):
    """Step C: the normalised target only if it wins by >= 1% relative on BOTH mean and median scaled pinball"""
    r, n = spec("raw", selected_cfg("raw", floor)), spec("norm", selected_cfg("norm", floor), floor=floor)
    return "norm" if selection.adopt_normalised(mean_sp(r), mean_spmed(r), mean_sp(n), mean_spmed(n)) else "raw"


def stage_groups(floor):
    var = adopted(floor)
    cfg = selected_cfg(var, floor)
    print(f"adopted variant: {var}, config {cfg}", flush=True)
    run([spec(var, cfg, a, floor) for a in ("base", "ev", "xs")], "D feature groups")
    for a in ("ev", "xs"):
        summarise(lambda a=a: print(f"group {a}: base {mean_sp(spec(var, cfg, 'base', floor)):.4f} with {mean_sp(spec(var, cfg, a, floor)):.4f} -> add: "
                                    f"{selection.add_group(mean_sp(spec(var, cfg, 'base', floor)), mean_sp(spec(var, cfg, a, floor)))}", flush=True))


def stage_sens(floor):
    """report-only sensitivities at the adopted variant: ids, year-ago flipped, future price removed, and the round-cap check (learning rate 0.1)"""
    var = adopted(floor)
    cfg = selected_cfg(var, floor)
    fa = final_arm(var, cfg, floor)
    run([spec(var, cfg, a, floor) for a in ("ids_all", "ids_noitem", "yearago_flip", "nofutprice")]
        + [spec(var, dict(cfg, learning_rate=0.1), fa, floor)], "S sensitivities (report-only)")


def final_arm(var, cfg, floor):
    base = mean_sp(spec(var, cfg, "base", floor))
    keep = [a for a in ("ev", "xs") if selection.add_group(base, mean_sp(spec(var, cfg, a, floor)))]
    return "+".join(keep) if keep else "base"


def stage_final(floor):
    var = adopted(floor)
    cfg = selected_cfg(var, floor)
    arm = final_arm(var, cfg, floor)
    print(f"final: variant {var}, config {cfg}, feature arm {arm}", flush=True)
    t0 = time.time()
    for idx, (fold, P) in enumerate(CELLS):
        ck = key(("final", var, cfg, arm, floor))
        if (ck, (fold, P)) in cache or idx % NSHARD != SHARD:
            continue
        fit, sel, cal, nscale = cell_data(fold, P)
        cols, y = cols_for(P, var, arm), fit[f"y_p{P}"]
        m = XGBQ(P, seed=0, columns=cols, normalize=(var == "norm"), floor=floor or 1 / 14, nthread=NTHREAD, **cfg).fit(fit, y)
        Qe, Qc = m.predict_quantiles(sel, ALPHAS), m.predict_quantiles(cal, ALPHAS)
        fl = floor or 1 / 14
        sc_e, sc_c = scale_of(sel, P, fl), scale_of(cal, P, fl)
        yc, sgc, sge = cal[f"y_p{P}"].to_numpy("float64"), cal.id.map(seg).to_numpy(), sel.id.map(seg).to_numpy()
        out = {"quantile_model": qrecord(Qe, sel, P, nscale)}
        # conformal variant: pooled by segment, service quantiles only, guard reported per segment and level
        offs, info = {a: {s: 0.0 for s in SEGS} for a in ALPHAS}, {}
        for a in SERVICE:
            sc = conformal.scores(yc, Qc[:, ALPHAS.index(a)], sc_c.to_numpy())
            for s in SEGS:
                n = int((sgc == s).sum())
                try:
                    offs[a][s] = conformal.offset(sc[sgc == s], a)
                    info[f"{a}_{s}"] = dict(n=n, thin=bool(conformal.thin(n, a)), offset=offs[a][s], ok=True)
                except conformal.InsufficientScores:
                    info[f"{a}_{s}"] = dict(n=n, thin=True, offset=None, ok=False)
        out["conformal"] = qrecord(conformal.apply_offsets(Qe, sge, sc_e.to_numpy(), offs, list(ALPHAS)), sel, P, nscale)
        out["conformal_info"] = info
        # benchmarks: the point policy's quantile, yhat + z sigma_seg scale, sigma pooled by segment on the calibration window
        yhat = {"B1_ma28": (REGISTRY["ma28"](P).predict(sel), REGISTRY["ma28"](P).predict(cal))}
        for nm, mk in (("B2_xgb", lambda: REGISTRY["xgb"](P, seed=0, columns=cols_for(P, var, arm), normalize=(var == "norm"), floor=fl, nthread=NTHREAD, **POINT_CFG)),
                       ("C_rf", lambda: REGISTRY["rf"](P, seed=0, columns=cols_for(P, var, arm), **RF_CFG))):
            pm = mk().fit(fit, y)
            yhat[nm] = (pm.predict(sel), pm.predict(cal))
        for nm, (pe, pc) in yhat.items():
            sig = quantiles.pooled_sigma((yc - pc) / sc_c.to_numpy(), sgc)
            sg_e = np.array([sig[s] for s in sge])
            Qb = np.column_stack([quantiles.point_policy_quantile(pe, sg_e, sc_e.to_numpy(), a) for a in ALPHAS])
            out[nm] = qrecord(Qb, sel, P, nscale)
            out[nm]["point_wape"] = metrics.wape(sel[f"y_p{P}"].to_numpy("float64"), pe)
        out["meta"] = dict(fold=fold, P=P, n_rounds=int(m.n_rounds_), hit_cap=bool(m.hit_cap_), crossing_share=m.crossing_share(sel))
        put(ck, (fold, P), out)
        save_cache()
        print(f"  final cell {fold} P={P} done ({time.time() - t0:.0f}s)", flush=True)


def flat(d, prefix=""):
    return {f"{prefix}{k}": v for k, v in d.items() if not isinstance(v, dict)}


def stage_report(floor):
    var = adopted(floor)
    cfg = selected_cfg(var, floor)
    arm = final_arm(var, cfg, floor)
    out = ROOT / "docs" / "results"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "ml", "tests"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    md = [f"# Phase 7 quantile forecasts on the tuning folds\n\nRun {date.today()} at code commit `{head}`{' (uncommitted changes in ml/ or tests/)' if dirty else ''}. "
          "Tuning folds only; nothing evaluated on the test window (touch log unchanged). One model version per fold and horizon (fit set ends at c - 84 days; "
          "12 Sunday evaluation origins; the reserved calibration window feeds sigma and conformal offsets only). Rules: design.md section 12, Phase 7 "
          "pre-registration and its amendment. Scaled pinball = per-series mean pinball / naive-P scale, averaged over the six quantiles; 'mean' and 'median' are over series. "
          "Means are over the 12 fold x horizon cells unless stated.\n"]
    # Step A / B
    fv = {f: mean_sp(spec("norm", FLOOR_CFG, floor=f)) for f in FLOORS}
    md.append("## Step A: scale floor (XGBoost quantile, depth 3 / min_child_weight 30, normalised)\n\n"
              + pd.DataFrame({"mean scaled pinball": {f"{f:.4f} units/day": round(v, 4) for f, v in fv.items()}}).to_markdown()
              + f"\n\nChosen: **{floor:.4f}** (rule: within 0.5% relative of the best is tied and goes to 1/14).\n")
    rows = []
    for v in ("raw", "norm"):
        for c in GRID:
            s = spec(v, c, floor=floor)
            cc = cells_of(s)
            rows.append(dict(variant=v, config=json.dumps(c), mean_sp=round(cc.sp_mean.mean(), 4), median_sp=round(cc.sp_median.mean(), 4),
                             F2_mean_sp=round(cc[cc.fold == "F2"].sp_mean.mean(), 4), rounds_hit_cap=int(cc.hit_cap.sum()),
                             selected="yes" if c == selected_cfg(v, floor) else ""))
    md.append("## Step B: config search (mean scaled pinball, lower is better)\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    # Step C
    r, n = spec("raw", selected_cfg("raw", floor)), spec("norm", selected_cfg("norm", floor), floor=floor)
    cmp = []
    for nm, s in (("raw", r), ("normalised", n)):
        cc = cells_of(s)
        cmp.append(dict(variant=nm, mean_sp=round(cc.sp_mean.mean(), 4), median_sp=round(cc.sp_median.mean(), 4), F2_mean_sp=round(cc[cc.fold == "F2"].sp_mean.mean(), 4),
                        sp_low=round(cc.sp_low.mean(), 4), sp_mid=round(cc.sp_mid.mean(), 4), sp_high=round(cc.sp_high.mean(), 4),
                        WAPE_of_median=round(cc.wape_median.mean(), 4), MASE_of_median=round(cc.mase_median_fc.mean(), 3)))
    md.append(f"## Step C: normalisation decision (both mean and median must win by >= 1% relative)\n\n" + pd.DataFrame(cmp).to_markdown(index=False)
              + f"\n\nRelative change of normalised vs raw: mean {mean_sp(n) / mean_sp(r) - 1:+.2%}, median {mean_spmed(n) / mean_spmed(r) - 1:+.2%}. **Adopted variant: {var}.**\n")
    # Step D
    d = []
    for a in ("base", "ev", "xs", "ids_all", "ids_noitem", "yearago_flip", "nofutprice", "final_lr0.1"):
        s = spec(var, dict(cfg, learning_rate=0.1), arm, floor) if a == "final_lr0.1" else spec(var, cfg, a, floor)
        if (key(s), CELLS[0]) not in cache:
            continue
        cc = cells_of(s)
        d.append(dict(arm=a, mean_sp=round(cc.sp_mean.mean(), 4), median_sp=round(cc.sp_median.mean(), 4), F2_mean_sp=round(cc[cc.fold == "F2"].sp_mean.mean(), 4),
                      vs_base=f"{cc.sp_mean.mean() / mean_sp(spec(var, cfg, 'base', floor)) - 1:+.2%}", WAPE_of_median=round(cc.wape_median.mean(), 4),
                      decision={"ev": "add" if selection.add_group(mean_sp(spec(var, cfg, 'base', floor)), mean_sp(s)) else "do not add",
                                "xs": "add" if selection.add_group(mean_sp(spec(var, cfg, 'base', floor)), mean_sp(s)) else "do not add"}.get(a, "report-only" if a != "base" else "")))
    md.append(f"## Step D: feature-group ablations at the adopted variant ({var}, {json.dumps(cfg)})\n\nGroups `ev` (30 named-event indicators) and `xs` (cross-store zero run) "
              "are added only if they lower mean scaled pinball by >= 0.5% relative; the other arms are report-only sensitivities (ids, year-ago flipped, future price removed).\n\n"
              + pd.DataFrame(d).to_markdown(index=False) + f"\n\nFinal shared feature set for both policies: base set of the {var} variant + `{arm}`.\n")
    # Step E
    ck = key(("final", var, cfg, arm, floor))
    fin = {c: cache[(ck, c)] for c in CELLS if (ck, c) in cache}
    if len(fin) == len(CELLS):
        def table(name):
            return pd.DataFrame([flat(fin[c][name]) | dict(fold=c[0], P=c[1]) for c in CELLS])
        names = {"quantile_model": "XGBoost quantile (headline)", "conformal": "XGBoost quantile + conformal (variant)", "B1_ma28": "B1: MA-28 + normal sigma",
                 "B2_xgb": "B2: XGBoost mean + normal sigma (point policy)", "C_rf": "C: RF mean + normal sigma (comparison only)"}
        summ, f2 = [], []
        for k, nm in names.items():
            t = table(k)
            summ.append(dict(method=nm, mean_sp=t.sp_mean.mean(), median_sp=t.sp_median.mean(), sp_low=t.sp_low.mean(), sp_mid=t.sp_mid.mean(), sp_high=t.sp_high.mean(),
                             **{f"sp_{a}": t[f"sp_{a}"].mean() for a in SERVICE}, interval80=t.interval80.mean()))
            tf = t[t.fold == "F2"]
            f2.append(dict(method=nm, mean_sp=tf.sp_mean.mean(), median_sp=tf.sp_median.mean(), **{f"sp_{a}": tf[f"sp_{a}"].mean() for a in SERVICE}))
        S = pd.DataFrame(summ).set_index("method")
        S["ratio_mean_to_B2"] = S.mean_sp / S.loc[names["B2_xgb"], "mean_sp"]
        S["ratio_mean_to_B1"] = S.mean_sp / S.loc[names["B1_ma28"], "mean_sp"]
        md.append("## Step E: scaled pinball against the point-policy benchmarks (mean over 12 cells; ratio below 1 is better)\n\n" + S.round(4).to_markdown()
                  + "\n\n### F2 (holiday fold) only\n\n" + pd.DataFrame(f2).set_index("method").round(4).to_markdown() + "\n")
        perP = pd.DataFrame([{**dict(P=c[1], fold=c[0]), **{nm: fin[c][k]["sp_mean"] for k, nm in names.items()}} for c in CELLS]).groupby("P").mean(numeric_only=True)
        md.append("### Mean scaled pinball by horizon\n\n" + perP.round(4).to_markdown() + "\n")
        for k, nm in ((("quantile_model"), "quantile model"), ("conformal", "quantile model + conformal"), ("B2_xgb", "B2 point policy (XGBoost mean + normal)")):
            t = table(k)
            rows = []
            for a in SERVICE:
                for scope, tt in (("all cells", t), ("F2 only", t[t.fold == "F2"])):
                    st = tt[f"status_{a}"].value_counts().to_dict()
                    rows.append(dict(alpha=a, scope=scope, cov_hi=round(tt[f"cov_hi_{a}"].mean(), 3), cov_lo=round(tt[f"cov_lo_{a}"].mean(), 3), tie=round(tt[f"tie_{a}"].mean(), 3),
                                     zero_share=round(tt[f"zero_share_{a}"].mean(), 3), raw_cov=round(tt[f"raw_cov_{a}"].mean(), 3),
                                     under=st.get("under", 0), consistent=st.get("consistent", 0), over=st.get("over", 0)))
            md.append(f"### Coverage of ceil(q): {nm} (status counts are over fold x horizon cells: {len(t)} all, {int((t.fold == 'F2').sum())} for F2)\n\n"
                      + pd.DataFrame(rows).to_markdown(index=False) + "\n")
        seg_rows = []
        for a in SERVICE:
            for s in SEGS:
                t = table("quantile_model")
                seg_rows.append(dict(alpha=a, segment=s, cov_hi=round(t[f"cov_hi_{a}_{s}"].mean(), 3), cov_lo=round(t[f"cov_lo_{a}_{s}"].mean(), 3)))
        md.append("### Coverage by velocity segment: quantile model (mean over cells)\n\n" + pd.DataFrame(seg_rows).to_markdown(index=False) + "\n")
        info = [dict(cell=f"{c[0]} P={c[1]}", key=k, **v) for c in CELLS for k, v in fin[c]["conformal_info"].items()]
        inf = pd.DataFrame(info)
        md.append("### Conformal: pooled scores per segment and level\n\n" + f"Cells x levels x segments: {len(inf)}; not calibrated (too few scores): {int((~inf.ok).sum())}; "
                  f"thin-sample warnings (n < 5/(1-alpha)): {int(inf.thin.sum())}; minimum n {int(inf.n.min())}, median n {int(inf.n.median())}.\n")
        metaf = pd.DataFrame([fin[c]["meta"] for c in CELLS])
        md.append(f"### Model diagnostics\n\nCells whose early stopping hit the 1,000-round cap: {int(metaf.hit_cap.sum())} of {len(metaf)}. "
                  f"Mean share of rows with crossing quantiles before sorting: {metaf.crossing_share.mean():.3f}.\n")
        pd.DataFrame([{**dict(method=k, fold=c[0], P=c[1]), **flat(fin[c][k])} for c in CELLS for k in names]).round(5).to_csv(out / "phase7_quantile_final_cells.csv", index=False)
    allr = pd.DataFrame([{**dict(variant=json.loads(k)[0], cfg=json.dumps(json.loads(k)[1]), arm=json.loads(k)[2], floor=json.loads(k)[3]), **flat(r)}
                         for (k, c), r in cache.items() if not k.startswith('["final"')])
    allr.round(5).to_csv(out / "phase7_quantile_all_fits.csv", index=False)
    (out / "phase7_quantile_tuning.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("floor", "all"):
        stage_floor()
    fl = chosen_floor()
    if what in ("search", "all"):
        stage_search(fl)
    if what in ("groups", "all"):
        stage_groups(fl)
    if what in ("final", "all"):
        stage_final(fl)
    if what in ("sens", "all"):
        stage_sens(fl)
    if what in ("report", "all"):
        stage_report(fl)
