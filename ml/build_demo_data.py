"""Builds the precomputed data of the demo from the committed result files (no model, no test-window data is read here).
    uv run python ml/build_demo_data.py             -> web/snapshot.json   (aggregated results only; committed; also what the database loader stores in `scenarios`)
    uv run python ml/build_demo_data.py quantiles   -> demo_private/quantiles.csv   (per-series quantile tables from the saved forecast tables; git-ignored and
                                                       NOT to be loaded into a public database until the M5 data-use terms are confirmed, see docs/design.md Phase 11)
The tables are read from the markdown result files by their headings, so a changed report layout fails loudly (see tests/test_demo_data.py)."""
import json
import re
import sys
from datetime import date

import pandas as pd

from config import PROCESSED, ROOT

RES = ROOT / "docs" / "results"
DATA_VERSION = "v1-2026-09-21"                      # must equal server/src/dataVersion.js
SCENARIOS = [
    ("primary", "Base case (lead time 3)", "phase10_simulator_primary", "The primary test-window run, made once. Lead time 3 days, weekly review, lost sales, 25 review dates."),
    ("L7", "Lead time 7", "phase10_sens_L7", "Sensitivity: lead time 7 days (protection interval 14 days)."),
    ("randlead", "Random lead time", "phase10_sens_randlead", "Sensitivity: each order's lead time Uniform{2, 3, 4}, ten draws pooled; policies plan with 3."),
    ("nofp", "No future-price inputs", "phase10_sens_nofp", "Sensitivity: models refit without the planned-window price features."),
    ("zerodemand", "Zero-demand cycles dropped", "phase10_diag_zerodemand", "Hindsight diagnostic, never a headline: cycles whose realised demand was zero are dropped from every policy alike."),
]
SEP = re.compile(r"^\|[\s:|-]+\|$")


def parse_tables(md):
    """[{title, header, rows}] for every markdown table; title = the last non-empty non-table line before it"""
    lines, out, title, i = md.splitlines(), [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and SEP.match(lines[i + 1].strip()):
            cells = lambda s: [c.strip() for c in s.strip().strip("|").split("|")]
            header, rows = cells(ln), []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            out.append(dict(title=title, header=header, rows=rows))
            continue
        if ln.strip() and not ln.startswith("|"):
            title = ln.strip()
        i += 1
    return out


def find(tables, prefix):
    hits = [t for t in tables if t["title"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one table titled {prefix!r}, found {len(hits)}")
    return hits[0]


def rows_of(t):
    return [dict(zip(t["header"], r)) for r in t["rows"]]


def scenario(sid, title, stem, blurb):
    md = (RES / f"{stem}.md").read_text(encoding="utf-8")
    tb = parse_tables(md)
    curves = pd.read_csv(RES / f"{stem}.csv")
    curves.columns = ["policy", "setting", "fill", "cycle_service", "inv"]
    out = dict(id=sid, title=title, blurb=blurb, source=f"docs/results/{stem}.md",
               curves=json.loads(curves.to_json(orient="records")),
               comparison_comparator_anchored=rows_of(find(tb, "**1b.")), comparison_quantile_anchored=rows_of(find(tb, "**1a.")),
               by_segment={s: rows_of(find(tb, f"**{s}**")) for s in ("low", "mid", "high")},
               service_gap=rows_of(find(tb, "## 3. Achieved cycle service")), cost=rows_of(find(tb, "## 7. Total cost")),
               sigma_forms=rows_of(find(tb, "## 6. Sigma form")))
    if any(t["title"].startswith("**peak**") for t in tb):
        out["by_period"] = {p: rows_of(find(tb, f"**{p}**")) for p in ("peak", "rest")}
    exp = re.findall(r"^- \*\*(HELD|DID NOT HOLD)\*\*: (.*)$", md, flags=re.M)
    if exp:
        out["expected_outcomes"] = [dict(held=h == "HELD", text=t) for h, t in exp]
    if sid == "primary":
        out["predictions"] = [ln for ln in md.splitlines() if ln.startswith(("Quantile policy: rest below target", "Cycle service at 0.99", "Verdict on the comparator"))]
    return out


def forecast_metrics():
    md = (RES / "phase10_test_forecast_metrics.md").read_text(encoding="utf-8")
    tb = parse_tables(md)
    h10 = md.split("## Horizon 14")[0]
    tb10 = parse_tables(h10)
    main = [t for t in tb10 if t["title"].startswith("### Quantile forecasts")][0]
    point = [t for t in tb10 if t["title"].startswith("### Point forecasts")][0]
    cmp_line = next(ln for ln in h10.splitlines() if ln.startswith("Quantile model against B3a"))
    return dict(source="docs/results/phase10_test_forecast_metrics.md", horizon=10, methods=rows_of(main), point=rows_of(point), against_b3a=cmp_line)


def null_tests():
    out = {}
    for k in ("test", "tuning"):
        tb = parse_tables((RES / f"phase11_null_test_{k}.md").read_text(encoding="utf-8"))
        out[k] = rows_of(tb[0])
    return out


def build():
    snap = dict(data_version=DATA_VERSION, built=str(date.today()),
                framing=("Every comparison leads with B3a, the post-hoc benchmark that keeps the same point forecast and takes its quantiles from empirical residuals: it is the fair benchmark. "
                         "B2 is the pre-specified normal-sigma point policy (its constant-variance assumption failed the Phase 9 check for high-velocity series)."),
                labels={"comparator-anchored": "the quantile policy's inventory interpolated at each comparator setting's own fill rate",
                        "quantile-anchored": "the comparator's inventory interpolated at the quantile policy's own fill rate (the pre-registered statistic)"},
                caveats=["Intervals come from a bootstrap over items: they reflect which items were sampled, not variation between periods (one 26-week test period).",
                         "The matched-inventory reduction is a descriptive comparison of two rules at equal fill rate under real demand, not evidence of forecast skill: a shuffled-demand null gives a positive reduction with no information (see null tests).",
                         "Cost is unresolved: the other policies' cost minima sit at their 0.80 grid edge and cannot be extended.",
                         "Historical replay under stated assumptions; no real stock, lead-time or cost data exist. Sales are not demand."],
                scenarios=[scenario(*s) for s in SCENARIOS], forecast_accuracy=forecast_metrics(), null_tests=null_tests())
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "snapshot.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print("wrote web/snapshot.json:", [s["id"] for s in snap["scenarios"]])


def quantiles():
    import versions
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["id", "date", "zero_run_91"]).set_index(["id", "date"]).zero_run_91
    frames = []
    for P, tag in ((10, "sim_test_v{v}_P10"), (14, "sim_test_v{v}_P14")):
        for v in versions.CUTOFFS:
            tab = pd.read_parquet(PROCESSED / (tag.format(v=v) + "_tab.parquet"))
            tab = tab[tab.date.isin(versions.use_dates(v))]
            tab = tab.assign(horizon=P, zero_run=(feats.reindex(pd.MultiIndex.from_arrays([tab.id, tab.date])).to_numpy() >= 14))
            frames.append(tab[["id", "date", "horizon", "q10", "q50", "q80", "q90", "q95", "q99", "zero_run"]].rename(columns={"id": "series", "date": "review_date"}))
    d = pd.concat(frames, ignore_index=True)
    (ROOT / "demo_private").mkdir(exist_ok=True)
    d.to_csv(ROOT / "demo_private" / "quantiles.csv", index=False)
    print("wrote demo_private/quantiles.csv:", len(d), "rows (git-ignored; do not publish)")


def features():
    """One row per (series, review_date) for every review date in the test window, tagged with the model version
    that actually serves it (versions.use_dates(v)), so live inference can cover every date, not just v4's."""
    import versions
    rows = []
    for v in (1, 2, 3, 4):
        name = f"v{v}_P10"
        meta = json.loads((ROOT / "models" / "serving" / f"{name}.meta.json").read_text(encoding="utf-8"))
        cols = meta["columns"]
        dates = versions.use_dates(v)
        feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["id", "date", *cols])
        feats = feats[feats.date.isin(dates)]
        rows.append(pd.DataFrame({
            "series": feats["id"], "review_date": feats["date"].dt.strftime("%Y-%m-%d"), "horizon": 10, "model": name,
            "payload": feats[cols].apply(lambda r: json.dumps({c: (None if pd.isna(v) else float(v)) for c, v in r.items()}), axis=1),
        }))
    out = pd.concat(rows, ignore_index=True)
    (ROOT / "demo_private").mkdir(exist_ok=True)
    out.to_csv(ROOT / "demo_private" / "features.csv", index=False)
    print("wrote demo_private/features.csv:", len(out), "rows across", len(rows), "model versions (git-ignored; do not publish)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    {"quantiles": quantiles, "features": features}.get(arg, build)()
