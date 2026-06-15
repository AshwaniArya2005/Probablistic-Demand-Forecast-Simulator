"""Forecast accuracy on the test window (touch-log row "Phase 10 test forecast metrics"): the primary run reported inventory and service results only. This reports, for the 25 counted review dates,
the quantile model and its benchmarks on the same forecast tables the replay used: WAPE of the point and median forecasts, scaled pinball over the six alphas (mean and median over series), the discrete-aware
coverage of ceil(q) at the four service levels, by velocity segment and by period, and an item-cluster bootstrap of the pinball difference against B3a. Horizon 10 (base case) and 14 (lead time 7).
These are the only test-window forecast metrics; resume and README numbers for forecast accuracy come from this file. Run: uv run python ml/test_forecast_metrics.py"""
from datetime import date

import numpy as np
import pandas as pd

import conformal
import metrics
import quantiles
import sim_run as sr
import versions
from config import HOLIDAY_END, PROCESSED, ROOT
from folds import touch_logged

ALPHAS = (0.10, 0.50, 0.80, 0.90, 0.95, 0.99)
SERVICE = (0.80, 0.90, 0.95, 0.99)
QN = ["q10", "q50", "q80", "q90", "q95", "q99"]
feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["id", "date", "mean_28", "sum_last_10", "sum_last_14", "y_p10", "y_p14"])
panel = pd.read_parquet(PROCESSED / "panel.parquet", columns=["id", "item_id", "segment"]).drop_duplicates("id").set_index("id")


def unit_rows(P, prefix, v):
    tab = pd.read_parquet(PROCESSED / f"{prefix}_v{v}_P{P}_tab.parquet")
    cal = pd.read_parquet(PROCESSED / f"{prefix}_v{v}_P{P}_cal.parquet")
    return tab, cal


def methods(tab, cal, P):
    """six-alpha quantile matrices for every method on the unit's rows (yhat, quantile model, B1, B2, B3a, B3c) and the point forecasts"""
    ma_cal = feats.set_index(["id", "date"]).mean_28.reindex(pd.MultiIndex.from_arrays([cal.id, cal.date])).to_numpy() * P
    sc_e, sc_c = tab.scale.to_numpy(), cal.scale.to_numpy()
    sg_e, sg_c = tab.segment.to_numpy(), cal.segment.to_numpy()
    y_c = cal.y.to_numpy()
    out = {"Quantile model": tab[QN].to_numpy()}

    def normal(pe, pc):
        sig = quantiles.pooled_sigma((y_c - pc) / sc_c, sg_c)
        return np.column_stack([quantiles.point_policy_quantile(pe, np.array([sig[g] for g in sg_e]), sc_e, a) for a in ALPHAS])

    def empirical(pe, pc):
        offs = {a: conformal.segment_offsets(conformal.scores(y_c, pc, sc_c), sg_c, a) for a in ALPHAS}
        return conformal.apply_offsets(np.repeat(pe[:, None], len(ALPHAS), axis=1), sg_e, sc_e, offs, list(ALPHAS))

    ma_e = tab.mu28.to_numpy() * P
    out["B3a (post-hoc): XGBoost mean + empirical residual quantiles"] = empirical(tab.yhat.to_numpy(), cal.yhat.to_numpy())
    out["B3c (post-hoc): MA-28 + empirical residual quantiles"] = empirical(ma_e, ma_cal)
    out["B2 (pre-specified): XGBoost mean + normal sigma"] = normal(tab.yhat.to_numpy(), cal.yhat.to_numpy())
    out["B1: MA-28 + normal sigma"] = normal(ma_e, ma_cal)
    return out, {"MA-28": ma_e, "XGBoost mean (normalised)": tab.yhat.to_numpy()}


def collect(P, prefix, L):
    dem = sr.load_market(sorted(pd.read_parquet(PROCESSED / f"{prefix}_v1_P{P}_tab.parquet").id.unique()))[0]
    days = dem.index
    cells = []
    for v in versions.CUTOFFS:
        counted = [d for d in versions.use_dates(v) if d >= versions.review_dates()[4]]
        if not counted:
            continue
        tab, cal = unit_rows(P, prefix, v)
        tab = tab[tab.date.isin(counted)].reset_index(drop=True)
        Q, pts = methods(tab, cal, P)
        y = np.array([dem[i].to_numpy()[days.get_loc(d) + 1:days.get_loc(d) + P + 1].sum() for i, d in zip(tab.id, tab.date)], float)
        nsc = metrics.naive_scale(feats.dropna(subset=[f"y_p{P}"]), P, before=versions.CUTOFFS[v])
        peak = (tab.date + pd.Timedelta(days=L + 1) <= HOLIDAY_END).to_numpy()
        cells.append(dict(v=v, tab=tab, y=y, Q=Q, pts=pts, nsc=nsc, peak=peak))
    return cells


def metric_rows(cells, mask=None):
    """one row per method: pooled over the given rows of every cell (mean over cells of each cell's value, like Phase 7)"""
    out = {}
    names = list(cells[0]["Q"])
    for nm in names:
        sp, wp, cov = [], [], {a: [] for a in SERVICE}
        for c in cells:
            m = np.ones(len(c["y"]), bool) if mask is None else mask(c)
            if not m.any():
                continue
            ids, y, Q = c["tab"].id[m], c["y"][m], c["Q"][nm][m]
            per = pd.concat([metrics.scaled_pinball_by_series(y, Q[:, j], a, ids, c["nsc"]) for j, a in enumerate(ALPHAS)], axis=1).mean(axis=1)
            sp.append((per.mean(), per.median()))
            wp.append(metrics.wape(y, Q[:, 1]))
            for a in SERVICE:
                cov[a].append(metrics.coverage_discrete(y, Q[:, ALPHAS.index(a)], a))
        r = {"scaled pinball, mean": np.mean([s[0] for s in sp]), "scaled pinball, median over series": np.mean([s[1] for s in sp]), "WAPE of the median": np.mean(wp)}
        for a in SERVICE:
            r[f"cov_hi {a}"] = np.mean([x["cov_hi"] for x in cov[a]])
            r[f"status {a}"] = "/".join(sorted({x["status"] for x in cov[a]}))
        out[nm] = r
    return pd.DataFrame(out).T


def point_rows(cells):
    rows = {}
    for nm in cells[0]["pts"]:
        rows[nm] = {"WAPE": np.mean([metrics.wape(c["y"], c["pts"][nm]) for c in cells])}
    rows["Quantile model median"] = {"WAPE": np.mean([metrics.wape(c["y"], c["Q"]["Quantile model"][:, 1]) for c in cells])}
    return pd.DataFrame(rows).T


def bootstrap(cells, other, B=10000, seed=0):
    """item-cluster bootstrap of (quantile model - other) in mean scaled pinball (mean over cells of the mean over series); negative favours the quantile model"""
    items = sorted(panel.item_id.unique())
    idx = {it: i for i, it in enumerate(items)}
    W = np.random.default_rng(seed).multinomial(len(items), np.full(len(items), 1 / len(items)), size=B).astype(float)
    diffs = []
    for c in cells:
        ids = c["tab"].id
        per = {}
        for nm in ("Quantile model", other):
            per[nm] = pd.concat([metrics.scaled_pinball_by_series(c["y"], c["Q"][nm][:, j], a, ids, c["nsc"]) for j, a in enumerate(ALPHAS)], axis=1).mean(axis=1)
        d = (per["Quantile model"] - per[other]).dropna()
        col = np.array([idx[panel.item_id[i]] for i in d.index])
        s = np.bincount(col, weights=d.to_numpy(), minlength=len(items))
        n = np.bincount(col, minlength=len(items)).astype(float)
        diffs.append((s, n))
    est = np.mean([s.sum() / n.sum() for s, n in diffs])
    b = np.mean([(W @ s) / (W @ n) for s, n in diffs], axis=0)
    return est, np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(df):
    return df.round(4).to_markdown()


HEAD = ("# Forecast accuracy on the test window (phase 10, made once)\n\n"
        "Run {today}. The primary replay reported inventory and service results only; this file is the test-window forecast accuracy, on the same forecast tables. **Only these numbers are quoted for forecast accuracy.**\n"
        "25 counted review dates (22 Nov 2015 to 8 May 2016) x 300 series; model versions v1 to v4 (v0 served only the warm-up); each version is one cell and values are means over cells, as in Phase 7. Scaled pinball is over "
        "the six alphas (0.10, 0.50, 0.80, 0.90, 0.95, 0.99) with each series scaled by its naive-P error before the version's cutoff. Coverage is of ceil(q) (cov_hi = share of demand at or below it; the status compares it "
        "with the nominal level, tolerance 0.03). B3a and B3c are post-hoc benchmarks; B2 is the pre-specified point policy; the framing rule applies. The item bootstrap reflects which items were sampled, not variation between periods.\n")
B3A = "B3a (post-hoc): XGBoost mean + empirical residual quantiles"
ORDER = [B3A, "Quantile model", "B3c (post-hoc): MA-28 + empirical residual quantiles", "B2 (pre-specified): XGBoost mean + normal sigma", "B1: MA-28 + normal sigma"]


def section(P, L, label):
    cells = collect(P, "sim_test", L)
    est, lo, hi = bootstrap(cells, B3A)
    est2, lo2, hi2 = bootstrap(cells, "B3c (post-hoc): MA-28 + empirical residual quantiles")
    wins = sum(metric_rows([c]).loc["Quantile model", "scaled pinball, mean"] < metric_rows([c]).loc[B3A, "scaled pinball, mean"] for c in cells)
    two = [B3A, "Quantile model"]
    parts = [f"\n## {label}\n", "### Quantile forecasts, all rows (B3a first)\n", fmt(metric_rows(cells).loc[ORDER]) + "\n",
             f"Quantile model against B3a: difference in mean scaled pinball {est:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}] over items; lower in {wins} of {len(cells)} cells (versions).\n"
             f"Quantile model against B3c: {est2:+.4f} [{lo2:+.4f}, {hi2:+.4f}].\n",
             "### Point forecasts (WAPE, mean over cells; the pipeline's mean forecast is the normalised-target XGBoost)\n", fmt(point_rows(cells)) + "\n",
             "### By period (peak: cycles starting on or before 2016-01-03; rest: later)\n", "**Peak**\n", fmt(metric_rows(cells, lambda c: c["peak"]).loc[two]) + "\n",
             "**Rest**\n", fmt(metric_rows(cells, lambda c: ~c["peak"]).loc[two]) + "\n", "### By velocity segment\n"]
    for s in ("high", "mid", "low"):
        parts += [f"**{s}**\n", fmt(metric_rows(cells, lambda c, s=s: (c["tab"].segment == s).to_numpy()).loc[two]) + "\n"]
    return "\n".join(parts)


def main():
    if not touch_logged("Phase 10 test forecast metrics"):
        raise SystemExit("refusing to run: no 'Phase 10 test forecast metrics' row in the touch log (docs/design.md section 14)")
    md = HEAD.format(today=date.today()) + section(10, 3, "Horizon 10 (base case, lead time 3)") + section(14, 7, "Horizon 14 (lead time 7 sensitivity)")
    (ROOT / "docs" / "results" / "phase10_test_forecast_metrics.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
