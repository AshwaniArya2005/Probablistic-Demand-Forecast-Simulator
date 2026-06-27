"""Pictorial versions of metrics that so far exist only as tables in docs/results/*.md and *.csv (report-only; no new numbers are computed, nothing is
scored, no touch-log row needed). Reads already-committed result files and writes PNGs to docs/figures/, in the same style as ml/eda.py's plots.
Categorical colors are the validated default palette (dataviz skill, references/palette.md), slots 1-5 in fixed order, checked with
scripts/validate_palette.js before use (two contrast WARNs on three of the five slots against the light surface; addressed here with a legend, an
in-panel B3a callout, and the fact that every figure's exact numbers are also in the markdown table next to it, the "relief" the skill requires).
Run: uv run python ml/metrics_figures.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ROOT

FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlelocation": "left", "font.size": 10,
                     "axes.edgecolor": "#52514e", "axes.titlesize": 11, "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # categorical slots 1-5, fixed order, validated (see module docstring)
GRAY = "#8a8a86"                                                    # reference/chance lines: neutral, not a data series


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")      # bbox_inches="tight" keeps the suptitle/long-title from being clipped
    plt.close(fig)
    print("wrote", FIGURES / name)


# ---------- A: test-window scaled pinball, B3a first, by horizon ----------
def pinball_figure():
    md = (ROOT / "docs" / "results" / "phase10_test_forecast_metrics.md").read_text(encoding="utf-8")
    import re
    rows = {}
    for h, block in re.findall(r"## Horizon (\d+).*?\n\n(\|.*?\n\n)", md, re.S):
        lines = [ln for ln in block.splitlines() if ln.startswith("|") and "---" not in ln][1:]
        rows[int(h)] = {ln.split("|")[1].strip(): float(ln.split("|")[2].strip()) for ln in lines}
    methods = ["Quantile model", "B3a (post-hoc): XGBoost mean + empirical residual quantiles", "B3c (post-hoc): MA-28 + empirical residual quantiles",
              "B2 (pre-specified): XGBoost mean + normal sigma", "B1: MA-28 + normal sigma"]
    short = ["Quantile\n(this model)", "B3a\n(fair benchmark)", "B3c\n(post-hoc)", "B2\n(pre-specified)", "B1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, h in zip(axes, (10, 14)):
        vals = [rows[h][m] for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, vals, color=SLOT, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
        ax.set_yticks(y, short, fontsize=9)
        ax.set_title(f"Horizon {h} days", loc="left")
        ax.set_xlabel("mean scaled pinball (lower is better)")
        ax.set_xlim(0, max(rows[10][m] for m in methods) * 1.18)
    fig.suptitle("Test-window forecast accuracy: B3a is the fair benchmark", x=0.01, ha="left", fontsize=11, y=1.03)
    save(fig, "test_window_scaled_pinball.png")


# ---------- B: coverage reliability, test window, both horizons ----------
def coverage_figure():
    md = (ROOT / "docs" / "results" / "phase10_test_forecast_metrics.md").read_text(encoding="utf-8")
    import re
    nominal = [0.80, 0.90, 0.95, 0.99]
    cov = {}
    for h, block in re.findall(r"## Horizon (\d+).*?\n\n(\|.*?\n\n)", md, re.S):
        lines = block.splitlines()
        header = [c.strip() for c in lines[0].split("|")]
        idx = [header.index(f"cov_hi {a:g}") for a in nominal]
        line = next(ln for ln in lines if ln.startswith("| Quantile model"))
        cells = [c.strip() for c in line.split("|")]
        cov[int(h)] = [float(cells[i]) for i in idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.78, 1.0], [0.78, 1.0], "--", color=GRAY, linewidth=1.5, label="perfect calibration")
    for (h, vals), color in zip(cov.items(), SLOT):
        ax.plot(nominal, vals, "-o", color=color, linewidth=2, markersize=8, label=f"horizon {h}")
    ax.set_xlabel("nominal service level"); ax.set_ylabel("achieved coverage of the ordered quantity")
    ax.set_title("Quantile model: achieved vs nominal coverage, test window", loc="left")
    ax.set_xlim(0.78, 1.0); ax.set_ylim(0.78, 1.0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "test_window_coverage_reliability.png")


# ---------- C/D: stockout ROC and PR curves (tuning folds, exploratory) ----------
def roc_pr_figures():
    C = pd.read_csv(ROOT / "docs" / "results" / "phase11_stockout_roc_curves.csv")
    cvals = sorted(C.c.unique())
    for curve, xlabel, ylabel, ref, fname, title in (
        ("roc", "false positive rate", "true positive rate", ([0, 1], [0, 1]), "stockout_roc_curves.png", "Stockout-risk ROC (tuning folds, exploratory)"),
        ("pr", "recall", "precision", None, "stockout_pr_curves.png", "Stockout-risk precision-recall (tuning folds, exploratory)"),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        if ref:
            ax.plot(*ref, "--", color=GRAY, linewidth=1.5, label="chance")
        for c, color in zip(cvals, SLOT):
            d = C[(C.c == c) & (C.curve == curve)].sort_values("x")
            ax.plot(d.x, d.y, "-", color=color, linewidth=2, label=f"c = {c:g}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="lower right" if curve == "roc" else "upper right")
        save(fig, fname)


if __name__ == "__main__":
    pinball_figure()
    coverage_figure()
    roc_pr_figures()
