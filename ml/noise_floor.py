"""Report-only (tuning folds, no test-window data): how close is the forecast error to a Poisson noise floor? For each Sunday origin the mean absolute deviation of a Poisson variable with mean equal to
the model's forecast, 2 e^-mu mu^(floor(mu)+1) / floor(mu)!, summed and divided by actual demand, next to the WAPE of the forecast and of MA-28. Real demand is overdispersed, so the floor is a lower bound
that is not reachable. Needs the parquet files of ml/policy_rows.py. Writes docs/results/phase9_noise_floor_tuning.md. Run: uv run python ml/noise_floor.py"""
import glob
from datetime import date

import numpy as np
import pandas as pd
from scipy.special import gammaln

from config import PROCESSED, ROOT

E = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "phase9_eval_*.parquet")))], ignore_index=True)


def poisson_mad(mu):
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    k = np.floor(mu)
    return np.exp(np.log(2) - mu + (k + 1) * np.log(mu) - gammaln(k + 1))


rows = []
for P, g in list(E.groupby("P")) + [("all", E)]:
    for seg, h in [("pooled", g)] + list(g.groupby("segment")):
        y = h.y.to_numpy()
        rows.append(dict(P=P, segment=seg, n=len(h), mean_demand=y.mean(), WAPE_forecast=np.abs(y - h.yhat.to_numpy()).sum() / y.sum(),
                         WAPE_MA28=np.abs(y - h.mu28.to_numpy() * h.P.to_numpy()).sum() / y.sum(), WAPE_poisson_floor=poisson_mad(h.yhat).sum() / y.sum()))
d = pd.DataFrame(rows).round(3)
md = f"""# Forecast error against a Poisson noise floor (tuning folds, report-only)

Run {date.today()}. `WAPE_forecast` is the normalised-target XGBoost mean forecast used by the B2 and B3a policies (Phase 9 rows: 12 Sunday origins x 4 folds x 300 series per horizon);
`WAPE_poisson_floor` is what a forecast equal to the truth would score if demand were Poisson around it. Real demand is overdispersed (promotions, SNAP days, stock-outs), so the floor cannot be reached.

{d.to_markdown(index=False)}
"""
(ROOT / "docs" / "results" / "phase9_noise_floor_tuning.md").write_text(md, encoding="utf-8")
print(md)
