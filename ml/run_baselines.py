"""Phase 5: naive, MA-28 and seasonal naive on the pre-registered tuning folds (ml/folds.py). Tuning folds only: folds.select refuses any
origin whose target window passes the tuning cutoff. Writes docs/results/phase5_baselines_tuning.{csv,md}.
Run: uv run python ml/run_baselines.py"""
import subprocess
from datetime import date

import pandas as pd

import metrics
from config import PROCESSED, ROOT
from features import PS
from folds import FOLDS, origins, select
from models import REGISTRY

feats = pd.read_parquet(PROCESSED / "features.parquet")
seg = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "segment"]).drop_duplicates("id").set_index("id").segment

rows, seg_rows = [], []
for fold in FOLDS:
    for P in PS:
        sel = select(feats, fold, P).reset_index(drop=True)
        y = sel[f"y_p{P}"].to_numpy("float64")
        scale = metrics.naive_scale(feats, P, before=origins(fold).min())       # only targets that ended before the fold began
        for name, cls in REGISTRY.items():
            m = cls(P)                                                               # baselines are stateless: nothing to fit
            p = m.predict(sel)
            mase, used, excl = metrics.mase(y, p, sel.id, scale)
            rows.append(dict(fold=fold, P=P, model=name, n=len(sel), wape=metrics.wape(y, p), mase=mase, mase_series=used, mase_excluded=excl,
                             mae=metrics.mae(y, p), rmse=metrics.rmse(y, p), share_y_le_forecast=metrics.coverage_onesided(y, p),
                             fallback_share=m.fallback_share(sel) if name == "snaive" else 0.0))
            for s in ("low", "mid", "high"):
                k = sel.id.map(seg).to_numpy() == s
                seg_rows.append(dict(fold=fold, P=P, model=name, segment=s, wape=metrics.wape(y[k], p[k]),
                                     share_y_le_forecast=metrics.coverage_onesided(y[k], p[k])))

res, by_seg = pd.DataFrame(rows), pd.DataFrame(seg_rows)
out = ROOT / "docs" / "results"
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / "phase5_baselines_tuning.csv", index=False)
by_seg.to_csv(out / "phase5_baselines_tuning_by_segment.csv", index=False)

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "ml", "tests"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
avg = res.groupby(["P", "model"])[["wape", "mase", "mae", "rmse", "share_y_le_forecast"]].mean().round(3)
avg_seg = by_seg.groupby(["P", "model", "segment"]).wape.mean().unstack().round(3)[["low", "mid", "high"]]
md = f"""# Phase 5 baselines on the tuning folds

Run {date.today()} at code commit `{head}`{' (uncommitted changes in ml/ or tests/)' if dirty else ''}. Tuning folds only (design section 12, `ml/folds.py`);
no test-window data was evaluated. Origins are Sundays: {len(FOLDS)} folds x 12 origins x 300 series = {res.n.iloc[0]} rows per fold per horizon.
Series-level detail: `phase5_baselines_tuning.csv`, `phase5_baselines_tuning_by_segment.csv`.

WAPE is pooled over the fold; MASE is the mean over series of MAE / the series' in-sample naive-P MAE (zero-scale series excluded and
counted in the CSV); `share_y_le_forecast` is the share of actuals at or below the point forecast, i.e. the quantile level the forecast
implicitly sits at (point baselines have no intervals). Averages below are simple means over the four folds.

## By fold

{res.round(3).to_markdown(index=False)}

## Mean over the four folds

{avg.to_markdown()}

## WAPE by velocity segment (mean over folds)

{avg_seg.to_markdown()}
"""
(out / "phase5_baselines_tuning.md").write_text(md, encoding="utf-8")
print(md)
