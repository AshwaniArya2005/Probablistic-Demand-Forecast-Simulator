"""Phase 6: Linear Regression, Random Forest and XGBoost point forecasts on the tuning folds. Every rule (folds, search space, selection,
floor, ablations, tie-breaks) is pre-registered in docs/design.md section 12 and committed before the first score.
One model version per fold and horizon: fit set = origins with s + P <= c_F - 84 days, evaluated on the fold's 12 Sunday origins.
Stages (resumable, results cached in data/processed/phase6_cache.json): floor -> search -> ablate -> report.
Run: uv run python ml/tune_point.py [floor|search|ablate|report|all]"""
import functools
import json
import subprocess
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

import metrics
import selection
import versions
from config import PROCESSED, ROOT
from features import PS, STATIC
from folds import FOLDS, assert_tuning_only, origins, select
from models import REGISTRY

CACHE = PROCESSED / "phase6_cache.json"
CELLS = [(f, P) for f in FOLDS for P in PS]
FLOORS = (1 / 56, 1 / 28, 1 / 14)
FLOOR_CFG = dict(max_depth=6, min_child_weight=30)                   # XGBoost config used to choose the floor
# grids listed simplest first (the tie-break order): RF shallower then larger leaf; XGB shallower then larger min_child_weight
GRIDS = {"lr": [{}],
         "rf": [dict(max_depth=d, min_samples_leaf=l) for d in (10, 16) for l in (60, 20)],
         "xgb": [dict(max_depth=d, min_child_weight=w) for d in (3, 6) for w in (30, 5)]}
ARMS = {"all": lambda P: (), "no_sum364": lambda P: (f"sum_364_{P}",), "no_item": lambda P: ("item_id",), "no_ids": lambda P: tuple(STATIC),
        "no_futprice": lambda P: (f"price_mean_rel_p{P}", f"price_min_rel_p{P}")}      # report-only (design section 4 known-prices ablation), no threshold

feats = pd.read_parquet(PROCESSED / "features.parquet")
seg = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "segment"]).drop_duplicates("id").set_index("id").segment
cache = {(e["k"], tuple(e["c"])): e["r"] for e in json.loads(CACHE.read_text())} if CACHE.exists() else {}


def save_cache():
    CACHE.write_text(json.dumps([{"k": k, "c": list(c), "r": r} for (k, c), r in cache.items()], default=float))


@functools.lru_cache(maxsize=4)
def cell_data(fold, P):
    c = origins(fold).min()
    fit = feats[versions.fit_mask(feats.date, c, P)]
    assert_tuning_only(fit.date, P)
    sel = select(feats, fold, P).reset_index(drop=True)
    return fit, sel, metrics.naive_scale(feats, P, before=c), sel.id.map(seg).to_numpy()


def score(model, fold, P):
    fit, sel, scale, sg = cell_data(fold, P)
    t = time.time()
    model.fit(fit, fit[f"y_p{P}"])
    fit_s = time.time() - t
    p, y = model.predict(sel), sel[f"y_p{P}"].to_numpy("float64")
    m, used, excl = metrics.mase(y, p, sel.id, scale)
    return dict(fold=fold, P=P, n_fit=len(fit), fit_seconds=round(fit_s, 1), wape=metrics.wape(y, p), mase=m,
                mase_median=metrics.mase_by_series(y, p, sel.id, scale).median(), mase_series=used, mase_excluded=excl,
                mae=metrics.mae(y, p), rmse=metrics.rmse(y, p), share_y_le_forecast=metrics.coverage_onesided(y, p),
                **{f"wape_{s}": metrics.wape(y[sg == s], p[sg == s]) for s in ("low", "mid", "high")})


def run(specs, label):
    """specs: list of (family, variant, cfg, arm, floor). Fits every uncached (spec, cell), cell-outermost so cell data is built once."""
    todo = [(s, c) for c in CELLS for s in specs if (json.dumps(s, sort_keys=True), c) not in cache]
    print(f"[{label}] {len(todo)} fits to run ({len(specs) * len(CELLS) - len(todo)} cached)", flush=True)
    t0, n = time.time(), 0
    for (fam, var, cfg, arm, floor), (fold, P) in todo:
        m = REGISTRY[fam](P, seed=0, normalize=(var == "norm"), floor=floor or 1 / 28, drop=ARMS[arm](P), **cfg)
        cache[(json.dumps((fam, var, cfg, arm, floor), sort_keys=True), (fold, P))] = score(m, fold, P)
        n += 1
        if n % 6 == 0 or n == len(todo):
            save_cache()
            print(f"  {n}/{len(todo)} fits, {time.time() - t0:.0f}s elapsed", flush=True)


def key(s):
    return json.dumps(s, sort_keys=True)


def cells_of(s):
    return pd.DataFrame([cache[(key(s), c)] for c in CELLS])


def mean_wape(s):
    return cells_of(s).wape.mean()


def spec(fam, var, cfg, arm="all", floor=None):
    return (fam, var, cfg, arm, floor if var == "norm" else None)


def choose_floor():
    w = {f: mean_wape(spec("xgb", "norm", FLOOR_CFG, floor=f)) for f in FLOORS}
    return selection.choose_floor(w), w


def stage_floor():
    run([spec("xgb", "norm", FLOOR_CFG, floor=f) for f in FLOORS], "floor")
    f, w = choose_floor()
    print("floor candidates (mean WAPE over 12 cells):", {round(k, 4): round(v, 4) for k, v in w.items()}, "-> chosen", round(f, 4), flush=True)


def selected(fam, var, floor):
    specs = [spec(fam, var, c, floor=floor) for c in GRIDS[fam]]
    return selection.simplest_within(specs, [mean_wape(s) for s in specs])


def stage_search(floor):
    run([spec(f, v, c, floor=floor) for f in GRIDS for v in ("raw", "norm") for c in GRIDS[f]], "search")
    for f in GRIDS:
        for v in ("raw", "norm"):
            print(f"selected {f}/{v}: {selected(f, v, floor)[2]}  mean WAPE {mean_wape(selected(f, v, floor)):.4f}", flush=True)


def stage_ablate(floor):
    sel = {(f, v): selected(f, v, floor) for f in GRIDS for v in ("raw", "norm")}
    specs = [(f, v, c, a, fl) for (f, v), (_, _, c, _, fl) in sel.items() for a in ("no_sum364",)]           # year-ago: all families
    specs += [(f, v, c, a, fl) for (f, v), (_, _, c, _, fl) in sel.items() if f in ("rf", "xgb") for a in ("no_item", "no_ids")]
    specs += [(f, v, c, "no_futprice", fl) for (f, v), (_, _, c, _, fl) in sel.items()]      # run after Phase 6 closed; report-only
    run(specs, "ablate")


def decisions(floor):
    out = []
    for v in ("raw", "norm"):
        b = spec("xgb", v, selected("xgb", v, floor)[2], floor=floor)
        a = (b[0], b[1], b[2], "no_sum364", b[4])
        keep = selection.keep_year_ago(mean_wape(b), mean_wape(a))
        out.append(dict(decision="year-ago sum_364_P", variant=v, with_feature=mean_wape(b), without=mean_wape(a), rule=f"keep iff XGBoost gain >= {selection.YEAR_AGO_GAIN}",
                        outcome="keep" if keep else "drop"))
        for f in ("rf", "xgb"):
            base = selected(f, v, floor)
            arms = {"all": base, "no_item": (f, v, base[2], "no_item", base[4]), "no_ids": (f, v, base[2], "no_ids", base[4])}
            w = {k: mean_wape(s) for k, s in arms.items()}
            out.append(dict(decision=f"ids ({f})", variant=v, with_feature=w["all"], without=w["no_ids"], no_item=w["no_item"],
                            rule=f"fewer ids preferred unless more ids gain >= {selection.ID_GAIN}", outcome=selection.choose_id_arm(w)))
    return pd.DataFrame(out)


def stage_report(floor):
    base = REGISTRY["ma28"]
    ma = pd.DataFrame([dict(family="ma28", variant="-", **score(base(P), f, P)) for f, P in CELLS])
    recs = []
    for (k, cell), r in cache.items():
        fam, var, cfg, arm, fl = json.loads(k)
        recs.append(dict(family=fam, variant=var, cfg=json.dumps(cfg), arm=arm, floor=fl, **r))
    allc = pd.DataFrame(recs)
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    allc.to_csv(out / "phase6_point_all_fits.csv", index=False)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "ml", "tests"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    M = ["wape", "mase", "mase_median", "share_y_le_forecast"]
    md = [f"# Phase 6 point forecasts on the tuning folds\n\nRun {date.today()} at code commit `{head}`{' (uncommitted changes in ml/ or tests/)' if dirty else ''}. "
          "Tuning folds only; no test-window data was evaluated (touch log unchanged). One model version per fold and horizon: fit set ends at c - 84 days, "
          "evaluated on the fold's 12 Sunday origins. All rules were pre-registered in design.md section 12 before scoring.\n\n"
          "**MA-28 is the primary benchmark.** `ratio` = model WAPE / MA-28 WAPE (below 1 is better). **No normalisation decision is made here**: both variants are "
          "reported side by side and carried into Phase 7, which decides on scaled pinball. Means are simple means over folds. "
          "Full per-fit results: `phase6_point_all_fits.csv`.\n"]
    fl, fw = choose_floor()
    md.append(f"## Scale floor (XGBoost depth 6, min_child_weight 30, normalised, mean WAPE over 12 cells)\n\n"
              + pd.Series({f"{k:.4f} units/day": round(v, 4) for k, v in fw.items()}).to_frame("mean WAPE").to_markdown() + f"\n\nChosen (provisional, Phase 7 may re-test): **{fl:.4f}**.\n")
    grid = []
    for f in GRIDS:
        for v in ("raw", "norm"):
            for c in GRIDS[f]:
                s = spec(f, v, c, floor=fl)
                grid.append(dict(family=f, variant=v, config=json.dumps(c), mean_wape=round(mean_wape(s), 4), F2_wape=round(cells_of(s).query("fold == 'F2'").wape.mean(), 4),
                                 selected="yes" if s == selected(f, v, fl) else ""))
    md.append("## Search results (mean WAPE over the 12 cells; F2 shown separately)\n\n" + pd.DataFrame(grid).to_markdown(index=False) + "\n")
    for v in ("raw", "norm"):
        rows = [ma] + [cells_of(selected(f, v, fl)).assign(family=f, variant=v) for f in GRIDS]
        d = pd.concat(rows, ignore_index=True)
        ref = d[d.family == "ma28"].groupby("P").wape.mean()
        blk = d.groupby(["P", "family"])[M].mean().round(3)
        blk["ratio_to_ma28"] = (d.groupby(["P", "family"]).wape.mean() / d.groupby(["P", "family"]).P.first().map(ref)).round(3)
        f2 = d[d.fold == "F2"].groupby(["P", "family"])[M].mean().round(3)
        f2["ratio_to_ma28"] = (d[d.fold == "F2"].groupby(["P", "family"]).wape.mean() / d[d.fold == "F2"].groupby(["P", "family"]).P.first().map(d[(d.family == "ma28") & (d.fold == "F2")].groupby("P").wape.mean())).round(3)
        sg = d.groupby(["P", "family"])[["wape_low", "wape_mid", "wape_high"]].mean().round(3)
        pf = d.pivot_table(index=["P", "family"], columns="fold", values="wape").round(3)
        cfgs = ", ".join(f"{f}: {selected(f, v, fl)[2] or 'no hyperparameters'}" for f in GRIDS)
        md.append(f"## Variant: {'raw target' if v == 'raw' else 'normalised target (floor ' + format(fl, '.4f') + ')'}\n\nSelected configs: {cfgs}.\n\n"
                  f"### Mean over the four folds\n\n{blk.to_markdown()}\n\n### F2 (holiday fold) only\n\n{f2.to_markdown()}\n\n"
                  f"### WAPE by velocity segment (mean over folds)\n\n{sg.to_markdown()}\n\n### WAPE per fold\n\n{pf.to_markdown()}\n")
    dec = decisions(fl)
    dec.round(4).to_csv(out / "phase6_point_ablation_decisions.csv", index=False)
    ab = []
    for f in GRIDS:
        for v in ("raw", "norm"):
            b = selected(f, v, fl)
            for a in ("all", "no_sum364", "no_item", "no_ids", "no_futprice"):
                if a in ("no_item", "no_ids") and f == "lr":
                    continue
                s = (b[0], b[1], b[2], a, b[4])
                if (key(s), CELLS[0]) not in cache:
                    continue
                c = cells_of(s)
                ab.append(dict(family=f, variant=v, arm=a, mean_wape=round(c.wape.mean(), 4), F2_wape=round(c[c.fold == "F2"].wape.mean(), 4),
                               mean_mase=round(c.mase.mean(), 3), mean_mase_median=round(c.mase_median.mean(), 3)))
    md.append("## Ablations at each selected config (mean over the 12 cells; F2 separately)\n\n" + pd.DataFrame(ab).to_markdown(index=False)
              + "\n\n### Decisions under the pre-registered rules\n\n" + dec.round(4).to_markdown(index=False) + "\n")
    (out / "phase6_point_tuning.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    floor = None
    if what in ("floor", "all"):
        stage_floor()
    floor = choose_floor()[0]
    if what in ("search", "all"):
        stage_search(floor)
    if what in ("ablate", "all"):
        stage_ablate(floor)
    if what in ("report", "all"):
        stage_report(floor)
