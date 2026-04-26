"""Report-only (added after Phase 6 closed, design.md section 12): for the frozen raw-target LR, RF and XGBoost configs against MA-28 on the
tuning folds: (1) the number of the 12 fold x horizon cells each model wins on WAPE, (2) an item-cluster bootstrap 95% interval on the pooled WAPE
difference (model minus MA-28). The three store-series of an item share its demand drivers, so items (100), not series (300), are resampled.
The interval reflects which items were sampled; it does not capture period-to-period variability (one set of folds).
Run: uv run python ml/bootstrap_point.py"""
import numpy as np
import pandas as pd

import tune_point as tp
from config import PROCESSED, ROOT
from features import PS
from models import REGISTRY

B, SEED = 10000, 0
floor = tp.choose_floor()[0]
rows = []
for fold, P in tp.CELLS:
    fit, sel, _, _ = tp.cell_data(fold, P)
    y = sel[f"y_p{P}"].to_numpy("float64")
    base = dict(fold=fold, P=P, item=sel.item_id.astype(str).to_numpy(), id=sel.id.to_numpy(), y=y)
    preds = {"ma28": REGISTRY["ma28"](P).predict(sel)}
    for fam in ("lr", "rf", "xgb"):
        _, var, cfg, arm, fl = tp.selected(fam, "raw", floor)
        preds[fam] = REGISTRY[fam](P, seed=0, **cfg).fit(fit, fit[f"y_p{P}"]).predict(sel)
    rows.append(pd.DataFrame({**base, **{f"p_{k}": v for k, v in preds.items()}}))
d = pd.concat(rows, ignore_index=True)
d.to_parquet(PROCESSED / "phase6_bootstrap_preds.parquet")

# reproducibility: the refit WAPEs must equal the logged ones
for fam in ("lr", "rf", "xgb", "ma28"):
    for (fold, P), g in d.groupby(["fold", "P"]):
        w = np.abs(g.y - g[f"p_{fam}"]).sum() / g.y.sum()
        logged = tp.cache[(tp.key(tp.selected(fam, "raw", floor)), (fold, P))]["wape"] if fam != "ma28" else None
        if logged is not None:
            assert abs(w - logged) < 1e-9, (fam, fold, P, w, logged)
print("refit WAPEs reproduce the logged Phase 6 values to 1e-9")

items = np.sort(d.item.unique())
rng = np.random.default_rng(SEED)
W = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=B).astype("float64")     # B x items, resampled counts


def diff_ci(g, fam):
    ae_m = np.abs(g.y - g[f"p_{fam}"]).groupby(g.item).sum().reindex(items, fill_value=0).to_numpy()
    ae_b = np.abs(g.y - g["p_ma28"]).groupby(g.item).sum().reindex(items, fill_value=0).to_numpy()
    ys = g.y.groupby(g.item).sum().reindex(items, fill_value=0).to_numpy()
    est = (ae_m - ae_b).sum() / ys.sum()
    boot = (W @ (ae_m - ae_b)) / (W @ ys)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


groups = {"all 12 cells": d, **{f"P = {P} (4 folds)": d[d.P == P] for P in PS}, "F2, all horizons": d[d.fold == "F2"],
          **{f"F2, P = {P}": d[(d.fold == "F2") & (d.P == P)] for P in (7, 14)}}
cellw = d.groupby(["fold", "P"]).apply(lambda g: pd.Series({f: np.abs(g.y - g[f"p_{f}"]).sum() / g.y.sum() for f in ("ma28", "lr", "rf", "xgb")}),
                                       include_groups=False)
out = []
for fam in ("lr", "rf", "xgb"):
    won = int((cellw[fam] < cellw["ma28"]).sum())
    for name, g in groups.items():
        est, lo, hi = diff_ci(g, fam)
        out.append(dict(family=fam, cells_won_of_12=won, group=name, wape_diff=est, ci_low=lo, ci_high=hi, excludes_zero="yes" if lo > 0 or hi < 0 else "no"))
res = pd.DataFrame(out)
res.round(4).to_csv(ROOT / "docs" / "results" / "phase6_point_bootstrap.csv", index=False)
md = ("# Phase 6: cells won and item-cluster bootstrap against MA-28\n\nRaw target, frozen configs, tuning folds only (report-only, added after Phase 6 closed). "
      f"`wape_diff` = pooled WAPE of the model minus MA-28 (negative = model better); 95% percentile interval from {B} resamples of the 100 items "
      "(each item's three store-series resampled together; seed 0). The interval covers item sampling only, not period-to-period variability "
      "(one set of folds), so it understates uncertainty about other periods. `cells_won_of_12` counts fold x horizon cells where the model's WAPE is below MA-28's.\n\n"
      + res.round(4).to_markdown(index=False) + "\n")
(ROOT / "docs" / "results" / "phase6_point_bootstrap.md").write_text(md, encoding="utf-8")
print(md)
