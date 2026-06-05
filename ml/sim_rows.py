"""Phase 10 forecast tables for the simulator, one "unit" = one model version. A unit refits the frozen Phase 7 quantile model and the XGBoost mean model on its fit set
(origins whose target ends by cutoff - 84 days), predicts at the review dates it serves, keeps the calibration-window rows (Sunday origins in [cutoff - 84, cutoff] with the
target observed by the cutoff) for sigma and offsets, and saves both models to the git-ignored models/ directory with a manifest line.
Development folds (P = 10; F2, F4):     sim_rows.py dev <i> <n>       -> data/processed/sim_dev_<fold>_P10_{tab,cal}.parquet
Test-window versions v0..v4 (P = 10):  sim_rows.py test <i> <n>      -> data/processed/sim_test_v<v>_P10_{tab,cal}.parquet   (refuses to run without the touch-log entry)
Sensitivity tables (each needs its own touch-log row): testP14 (horizon 14, lead time 7) -> sim_test_v<v>_P14_*; testnofp (no future-price inputs, the Phase 7 "nofutprice" arm)
-> sim_test_nofp_v<v>_P10_*. Sharded like policy_rows.py over the units."""
import hashlib
import json
import subprocess

import pandas as pd

import tune_quantile as tq
import versions
from config import ROOT
from folds import TOUCH_KEY, assert_tuning_only, origins, touch_logged
from learned import XGBQ
from models import REGISTRY

P = 10
VAR, CFG = "norm", dict(max_depth=6, min_child_weight=30)
MODELS = ROOT / "models"


def warmup_origins(first):
    return pd.date_range(first - pd.Timedelta(weeks=4), first - pd.Timedelta(weeks=1), freq="7D")


def manifest(path, **kw):
    kw["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    kw["file"] = path.name
    kw["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    with open(MODELS / "manifest.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


def build_unit(name, cutoff, dates, floor, P=P, arm="base"):
    """fit the unit's models at `cutoff` and return (tab, cal); saves the models. `dates`: the review dates it serves (plus warm-up dates for version 0 / a fold)."""
    feats = tq.feats
    fit = feats[versions.fit_mask(feats.date, cutoff, P)]
    cal = feats[versions.calibration_sundays(feats.date, cutoff, P)].reset_index(drop=True)
    rows = feats[feats.date.isin(dates)].reset_index(drop=True)
    assert rows.groupby("date").id.nunique().eq(300).all() and rows.date.nunique() == len(dates), "every series at every review date"
    cols, y = tq.cols_for(P, VAR, arm), fit[f"y_p{P}"]
    qm = XGBQ(P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **CFG).fit(fit, y)
    mm = REGISTRY["xgb"](P, seed=0, columns=cols, normalize=True, floor=floor, nthread=tq.NTHREAD, **tq.POINT_CFG).fit(fit, y)
    Q = qm.predict_quantiles(rows, tq.ALPHAS)
    tab = pd.DataFrame({"id": rows.id.to_numpy(), "date": rows.date.to_numpy(), "segment": rows.id.map(tq.seg).to_numpy(), "mu28": rows.mean_28.to_numpy(),
                        "scale": tq.scale_of(rows, P, floor).to_numpy(), "yhat": mm.predict(rows), **{f"q{round(a * 100)}": Q[:, j] for j, a in enumerate(tq.ALPHAS)}})
    cal_t = pd.DataFrame({"id": cal.id.to_numpy(), "date": cal.date.to_numpy(), "segment": cal.id.map(tq.seg).to_numpy(), "y": cal[f"y_p{P}"].to_numpy(),
                          "yhat": mm.predict(cal), "scale": tq.scale_of(cal, P, floor).to_numpy()})
    for role, m in (("quantile", qm), ("mean", mm)):
        path = MODELS / f"{name}_P{P}_{role}.json"
        m.save(path)
        manifest(path, role=role, unit=name, arm=arm, P=P, cutoff=str(cutoff.date()), fit_end=str(versions.fit_end(cutoff).date()), floor=float(floor), n_columns=len(cols), columns=cols)
    return tab, cal_t


if __name__ == "__main__":
    import sys
    kind = sys.argv[1]
    MODELS.mkdir(exist_ok=True)
    floor = tq.chosen_floor()
    if kind == "dev":
        units = []
        for fold in ("F2", "F4"):
            first = origins(fold).min()
            units.append((f"dev_{fold}", f"sim_dev_{fold}_P{P}", first, warmup_origins(first).append(origins(fold))))
    elif kind in ("test", "testP14", "testnofp"):
        key = {"test": TOUCH_KEY, "testP14": "Phase 10 sensitivity L7", "testnofp": "Phase 10 sensitivity nofp"}[kind]
        if not touch_logged(key):
            raise SystemExit(f"refusing to build test-window forecast tables: no '{key}' row in the touch log (docs/design.md section 14)")
        PP, arm = (14 if kind == "testP14" else P), ("nofutprice" if kind == "testnofp" else "base")
        pre = "sim_test_nofp" if kind == "testnofp" else "sim_test"
        units = [(f"{pre}_v{v}" if kind == "testnofp" else f"test_v{v}", f"{pre}_v{v}_P{PP}", c, versions.use_dates(v)) for v, c in versions.CUTOFFS.items()]
    else:
        raise SystemExit("usage: sim_rows.py dev|test <i> <n>")
    for idx, (name, stem, cutoff, dates) in enumerate(units):
        if idx % tq.NSHARD != tq.SHARD:
            continue
        if kind == "dev":
            assert_tuning_only(dates, P)
        tab, cal_t = build_unit(name, cutoff, pd.DatetimeIndex(dates), floor, *(((PP, arm)) if kind != "dev" else (P, "base")))
        tab.to_parquet(tq.PROCESSED / f"{stem}_tab.parquet")
        cal_t.to_parquet(tq.PROCESSED / f"{stem}_cal.parquet")
        print(f"{name} done", flush=True)
