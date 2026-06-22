"""Retrain one cheap frozen version from scratch (development fold F4, horizon 10; tuning data only) and compare its forecasts with (a) the forecast table the results were built from and (b) the saved
model file that ml/model_hashes.py verifies, within a stated tolerance. Same recipe, seed and thread count as ml/sim_rows.py. Nothing is saved to models/. Needs ~5 minutes.
    uv run python ml/retrain_check.py                     Writes docs/results/phase11_retrain_check.md"""
import hashlib
import subprocess
from datetime import date

import numpy as np
import pandas as pd

import tune_quantile as tq
import versions
from config import PROCESSED, ROOT
from folds import assert_tuning_only, origins
from learned import XGBQ
from models import REGISTRY

P, FOLD, TOL = 10, "F4", 1e-4                       # tolerance in units of demand (forecasts are float32 tree sums)
floor = tq.chosen_floor()
first = origins(FOLD).min()
assert_tuning_only(origins(FOLD), P)
feats = tq.feats
fit = feats[versions.fit_mask(feats.date, first, P)]
cols, y = tq.cols_for(P, "norm", "base"), fit[f"y_p{P}"]
tab = pd.read_parquet(PROCESSED / f"sim_dev_{FOLD}_P{P}_tab.parquet")
tab = tab[tab.date.isin(origins(FOLD))].reset_index(drop=True)
rows = feats.set_index(["id", "date"]).loc[list(zip(tab.id, tab.date))].reset_index()
Q_cols = ["q10", "q50", "q80", "q90", "q95", "q99"]

qm = XGBQ(P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, max_depth=6, min_child_weight=30).fit(fit, y)
mm = REGISTRY["xgb"](P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **tq.POINT_CFG).fit(fit, y)
new_q, new_m = qm.predict_quantiles(rows, XGBQ.ALPHAS), mm.predict(rows)
saved_q = XGBQ.load(ROOT / "models" / f"dev_{FOLD}_P{P}_quantile.json")
saved_file_q = saved_q.predict_quantiles(rows, XGBQ.ALPHAS)
d_table_q, d_table_m = np.abs(new_q - tab[Q_cols].to_numpy()).max(), np.abs(new_m - tab.yhat.to_numpy()).max()
d_saved_q = np.abs(new_q - saved_file_q).max()
trees_new, trees_saved = len(qm.model.get_dump()), len(saved_q.model.get_dump())
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
ok = max(d_table_q, d_table_m, d_saved_q) <= TOL and trees_new == trees_saved
md = f"""# Retraining one frozen version from scratch (tuning data only)

Run {date.today()} at commit `{head}`. Development fold {FOLD}, horizon {P} (cutoff {first.date()}, fit set ending {versions.fit_end(first).date()}), the frozen Phase 7 recipe (normalised target, floor {floor:.4f}, quantile XGBoost
max_depth 6 / min_child_weight 30, mean XGBoost max_depth 3 / min_child_weight 30), seed 0, {tq.NTHREAD} threads. Compared on the {len(tab)} review-origin rows of the fold.

| comparison | maximum absolute difference (units of demand) | within tolerance {TOL} |
|---|---:|---|
| new quantile fit vs the quantile forecasts in the forecast table the results were built from | {d_table_q:.3g} | {'yes' if d_table_q <= TOL else 'NO'} |
| new mean fit vs the mean forecasts in that table | {d_table_m:.3g} | {'yes' if d_table_m <= TOL else 'NO'} |
| new quantile fit vs the saved model file (hash recorded in `models/manifest.jsonl`) | {d_saved_q:.3g} | {'yes' if d_saved_q <= TOL else 'NO'} |
| trees in the new quantile model vs the saved one | {trees_new} vs {trees_saved} | {'yes' if trees_new == trees_saved else 'NO'} |

**Result: {'the retrained version reproduces the frozen forecasts within the stated tolerance' if ok else 'DIFFERENCE BEYOND TOLERANCE, see the table'}.** Scope: one cheap version on the same machine, thread count and library versions; it does not show
cross-machine bit-reproducibility or that every version reproduces, and the test-window versions were not retrained (that would need its own touch-log entry).
"""
(ROOT / "docs" / "results" / "phase11_retrain_check.md").write_text(md, encoding="utf-8")
print(md)
