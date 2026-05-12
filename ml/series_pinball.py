"""Per-series scaled pinball for the frozen Phase 7 quantile model and every benchmark, including the post-hoc B3 variants (point forecast plus empirical
residual quantiles from the calibration window; design.md section 12, 2026-09-21). Tuning folds only. Refits the frozen configs and asserts that every
method with a cached Phase 7 counterpart reproduces its cell value to 1e-9. Sharded like tune_quantile: series_pinball.py series <i> <n>.
Writes data/processed/phase7_series_<i>.parquet and phase7_b3_<i>.json (git-ignored). Analysis: ml/bootstrap_quantile.py."""
import json

import numpy as np
import pandas as pd

import conformal
import metrics
import quantiles
import tune_quantile as tq
from learned import XGBQ
from models import REGISTRY

ALPHAS = tq.ALPHAS
VAR, CFG = "norm", dict(max_depth=6, min_child_weight=30)
COUNTERPART = {"quantile_model": ("final", "quantile_model"), "B1_normal": ("final", "B1_ma28"), "B2_normal": ("final", "B2_xgb"),
               "B2raw_normal": ("final_b", "B2_raw")}


def empirical(pe, pc, sc_e, sc_c, y_c, sg_e, sg_c):
    """B3: yhat + pooled-by-segment empirical residual offsets (conformal rule and guard), all six alphas"""
    offs = {}
    for a in ALPHAS:
        s = conformal.scores(y_c, pc, sc_c)
        offs[a] = conformal.segment_offsets(s, sg_c, a)
    return conformal.apply_offsets(np.repeat(pe[:, None], len(ALPHAS), axis=1), sg_e, sc_e, offs, list(ALPHAS))


def normal(pe, pc, sc_e, sc_c, y_c, sg_e, sg_c):
    sig = quantiles.pooled_sigma((y_c - pc) / sc_c, sg_c)
    s = np.array([sig[g] for g in sg_e])
    return np.column_stack([quantiles.point_policy_quantile(pe, s, sc_e, a) for a in ALPHAS])


if __name__ == "__main__":
    floor = tq.chosen_floor()
    rows, recs = [], {}
    for idx, (fold, P) in enumerate(tq.CELLS):
        if idx % tq.NSHARD != tq.SHARD:
            continue
        fit, sel, cal, nscale = tq.cell_data(fold, P)
        cols, y = tq.cols_for(P, VAR, "base"), fit[f"y_p{P}"]
        y_e, y_c = sel[f"y_p{P}"].to_numpy("float64"), cal[f"y_p{P}"].to_numpy("float64")
        sc_e, sc_c = tq.scale_of(sel, P, floor).to_numpy(), tq.scale_of(cal, P, floor).to_numpy()
        sg_e, sg_c = sel.id.map(tq.seg).to_numpy(), cal.id.map(tq.seg).to_numpy()
        item_of = sel.drop_duplicates("id").set_index("id").item_id.astype(str)
        qm = XGBQ(P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **CFG).fit(fit, y)
        point = {"ma28": (REGISTRY["ma28"](P).predict(sel), REGISTRY["ma28"](P).predict(cal))}
        for nm, m in (("xgb", REGISTRY["xgb"](P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **tq.POINT_CFG)),
                      ("raw_xgb", REGISTRY["xgb"](P, seed=0, nthread=tq.NTHREAD, **tq.POINT_CFG))):
            m.fit(fit, y)
            point[nm] = (m.predict(sel), m.predict(cal))
        methods = {"quantile_model": qm.predict_quantiles(sel, ALPHAS)}
        for nm, label in (("ma28", "B1_normal"), ("xgb", "B2_normal"), ("raw_xgb", "B2raw_normal")):
            methods[label] = normal(*point[nm], sc_e, sc_c, y_c, sg_e, sg_c)
        for nm, label in (("xgb", "B3a_xgb"), ("raw_xgb", "B3b_raw_xgb"), ("ma28", "B3c_ma28")):
            methods[label] = empirical(*point[nm], sc_e, sc_c, y_c, sg_e, sg_c)
        for label, Q in methods.items():
            per = pd.concat([metrics.scaled_pinball_by_series(y_e, Q[:, j], a, sel.id, nscale).rename(f"sp_{a}") for j, a in enumerate(ALPHAS)], axis=1)
            per["sp_mean6"] = per.mean(axis=1)
            per = per.reset_index(names="id").assign(fold=fold, P=P, method=label, item=lambda d: d.id.map(item_of))
            per["segment"] = per.id.map(tq.seg)
            rows.append(per)
            if label in COUNTERPART:
                grp, key = COUNTERPART[label]
                ck = tq.key(("final", VAR, CFG, "base", floor)) if grp == "final" else tq.key(("final_b", "phase6_raw_xgb", floor))
                cached = tq.cache[(ck, (fold, P))][key]["sp_mean"]
                assert abs(per.sp_mean6.mean() - cached) < 1e-9, (label, fold, P, per.sp_mean6.mean(), cached)
            if label.startswith("B3"):
                recs[f"{label}|{fold}|{P}"] = tq.qrecord(Q, sel, P, nscale)
        print(f"cell {fold} P={P} done; reproduced the cached Phase 7 values", flush=True)
    out = tq.PROCESSED
    pd.concat(rows, ignore_index=True).to_parquet(out / f"phase7_series_{tq.SHARD}.parquet")
    (out / f"phase7_b3_{tq.SHARD}.json").write_text(json.dumps(recs, default=float))
