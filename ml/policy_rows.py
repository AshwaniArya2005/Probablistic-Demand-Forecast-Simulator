"""Phase 9 data: for every tuning fold x horizon cell, refit the frozen Phase 7 quantile model and the XGBoost mean model (adopted variant, shared feature
set) and save per-row values for ml/policy_check.py: the evaluation rows (quantiles, mean forecast, scale, mu28, actuals) and the calibration rows (actuals,
mean forecast, scale, segment). Tuning folds only. Sharded: policy_rows.py rows <i> <n>. Writes data/processed/phase9_eval_<i>.parquet / phase9_cal_<i>.parquet."""
import numpy as np
import pandas as pd

import tune_quantile as tq
from learned import XGBQ
from models import REGISTRY

ALPHAS = tq.ALPHAS
VAR, CFG = "norm", dict(max_depth=6, min_child_weight=30)

if __name__ == "__main__":
    floor = tq.chosen_floor()
    ev, ca = [], []
    for idx, (fold, P) in enumerate(tq.CELLS):
        if idx % tq.NSHARD != tq.SHARD:
            continue
        fit, sel, cal, nscale = tq.cell_data(fold, P)
        cols, y = tq.cols_for(P, VAR, "base"), fit[f"y_p{P}"]
        qm = XGBQ(P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **CFG).fit(fit, y)
        mm = REGISTRY["xgb"](P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **tq.POINT_CFG).fit(fit, y)
        Q = qm.predict_quantiles(sel, ALPHAS)
        cached = tq.cache[(tq.key(("final", VAR, CFG, "base", floor)), (fold, P))]["quantile_model"]["sp_mean"]
        assert abs(tq.qrecord(Q, sel, P, nscale)["sp_mean"] - cached) < 1e-9, "the refit must reproduce the cached Phase 7 cell value"
        ev.append(pd.DataFrame({"fold": fold, "P": P, "id": sel.id.to_numpy(), "date": sel.date.to_numpy(), "segment": sel.id.map(tq.seg).to_numpy(),
                                "y": sel[f"y_p{P}"].to_numpy(), "mu28": sel.mean_28.to_numpy(), "scale": tq.scale_of(sel, P, floor).to_numpy(),
                                "yhat": mm.predict(sel), **{f"q{round(a * 100)}": Q[:, j] for j, a in enumerate(ALPHAS)}}))
        ca.append(pd.DataFrame({"fold": fold, "P": P, "id": cal.id.to_numpy(), "date": cal.date.to_numpy(), "segment": cal.id.map(tq.seg).to_numpy(),
                                "y": cal[f"y_p{P}"].to_numpy(), "yhat": mm.predict(cal), "scale": tq.scale_of(cal, P, floor).to_numpy()}))
        print(f"cell {fold} P={P} done (reproduced the cached Phase 7 value)", flush=True)
    pd.concat(ev, ignore_index=True).to_parquet(tq.PROCESSED / f"phase9_eval_{tq.SHARD}.parquet")
    pd.concat(ca, ignore_index=True).to_parquet(tq.PROCESSED / f"phase9_cal_{tq.SHARD}.parquet")
