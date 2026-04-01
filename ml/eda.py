"""Phase 3 EDA on data/processed/panel.parquet: prints the findings, writes docs/figures/*.png.
Anything that feeds a decision (segments, thresholds) uses data <= TUNING_CUTOFF only; whole-history views are labelled as such.
Run: uv run python ml/eda.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *

panel = pd.read_parquet(PROCESSED / "panel.parquet")
opn = panel[~panel.closed]
train = opn[opn.date <= TUNING_CUTOFF]
seg = panel.groupby("id").segment.first()
SEG_COLOR = {"low": "#2a78d6", "mid": "#eb6834", "high": "#1baf7a"}   # categorical slots 1-3 (all-pairs safe)
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlelocation": "left", "font.size": 10,
                     "axes.edgecolor": "#52514e", "axes.titlesize": 11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


print(f"series {panel.id.nunique()}, items {panel.item_id.nunique()}, rows {len(panel)}, {panel.date.min().date()}..{panel.date.max().date()}")
print("series by segment (label from data <= cutoff):", seg.value_counts().to_dict())
print("series by dept:", panel.groupby("id").dept_id.first().value_counts().to_dict())

# ---- 1. sparsity and intermittency (training window) ----
print("\n== intermittency, training window, open days ==")
g = train.groupby("id").sales
adi = g.size() / g.apply(lambda s: (s > 0).sum())
nz = train[train.sales > 0].groupby("id").sales
cv2 = (nz.std() / nz.mean()) ** 2
cls = np.select([(adi < 1.32) & (cv2 < .49), (adi < 1.32), (cv2 < .49)], ["smooth", "erratic", "intermittent"], "lumpy")
sb = pd.crosstab(seg.reindex(adi.index), cls)
print("Syntetos-Boylan class by segment (ADI 1.32, CV2 0.49):\n", sb.reindex(["low", "mid", "high"]).to_string())
print("daily zero share by segment:", train.assign(z=train.sales == 0).groupby("segment", observed=True).z.mean().round(3).to_dict())
print("daily mean units by segment:", train.groupby("segment", observed=True).sales.mean().round(2).to_dict())

# ---- 2. why the target is a P-day sum: zero share of rolling sums ----
wide = panel.pivot(index="date", columns="id", values="sales")           # NaN before launch
wide = wide[wide.index <= TUNING_CUTOFF]
share = {}
for P in (1, 7, 14):
    r = wide.rolling(P, min_periods=P).sum().where(wide.notna().rolling(P, min_periods=P).sum() == P)
    share[P] = (r == 0).where(r.notna()).mean().groupby(seg, observed=True).mean()  # per-series zero share, averaged by segment
share = pd.DataFrame(share).reindex(["low", "mid", "high"])
print("\nshare of zero-demand windows by horizon (rows=segment, cols=P days):\n", share.round(3).to_string())
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.26
for i, s in enumerate(share.index):
    ax.bar(np.arange(3) + (i - 1) * (w + 0.02), share.loc[s], w, color=SEG_COLOR[s], label=s)
    for x, v in zip(np.arange(3) + (i - 1) * (w + 0.02), share.loc[s]):
        ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color="#52514e")
ax.set_xticks(range(3), ["1 day", "7 days", "14 days"])
ax.set_ylabel("share of windows with zero sales")
ax.set_title("Daily demand is often zero; P-day sums much less so\n(series-averaged, data to 2015-08-30)", fontsize=10)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(title="velocity segment", frameon=False)
ax.grid(axis="x", visible=False)
save(fig, "zero_share_by_horizon.png")

# ---- 3. intermittency scatter ----
fig, ax = plt.subplots(figsize=(6.4, 4))
for s in ("low", "mid", "high"):
    i = seg.index[seg == s].intersection(adi.index)
    ax.scatter(adi[i], cv2[i], s=16, color=SEG_COLOR[s], edgecolor="white", linewidth=0.6, label=s)
ax.axvline(1.32, color="#52514e", lw=0.8, ls="--")
ax.axhline(0.49, color="#52514e", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.set_xlabel("average demand interval (days between sales, log scale)")
ax.set_ylabel("CV² of non-zero sizes")
ax.set_title("Most low-velocity series are intermittent or lumpy", fontsize=10)
ax.legend(title="velocity segment", frameon=False)
save(fig, "intermittency.png")

# ---- 4. level, seasonality, regime ----
daily = opn.groupby("date").sales.mean()
weekly = daily.resample("W-SUN").mean()
full = panel.groupby("id").date.min()
stable = panel[panel.id.map(full) <= "2011-06-30"]                 # on sale throughout: no launch composition effects
stable = stable[~stable.closed]
sd = stable.groupby("date").sales.mean()
print("\n== holiday-peak regime (series on sale since mid-2011; whole history, descriptive only) ==")
for y in range(2011, 2016):
    peak = sd[f"{y}-11-23":f"{y + 1}-01-03"].mean()
    base = sd[f"{y}-09-01":f"{y}-10-31"].mean()
    print(f"  {y}-11-23..{y + 1}-01-03: mean daily units {peak:.2f} vs Sep-Oct {base:.2f}  (x{peak / base:.2f})")
fig, ax = plt.subplots(figsize=(8, 3.6))
sw = sd.resample("W-SUN").mean()
ax.plot(sw.index, sw.values, color="#2a78d6", lw=1.4)
ax.axvspan(TEST_START, HOLIDAY_END, color="#eda100", alpha=0.22, lw=0)
ax.axvspan(HOLIDAY_END, LAST_SALES_DATE, color="#52514e", alpha=0.08, lw=0)
ax.axvline(TUNING_CUTOFF, color="#52514e", lw=0.8, ls="--")
ax.text(TUNING_CUTOFF - pd.Timedelta(days=10), sw.max() * 0.05, "tuning cutoff", fontsize=8, ha="right", color="#52514e")
ax.text(LAST_SALES_DATE, sw.max() * 1.03, "test: holiday peak, then rest", fontsize=8, ha="right", color="#52514e")
ax.set_ylim(0, sw.max() * 1.12)
ax.set_ylabel("mean units per series-day (weekly)")
ax.set_title("Series on sale since mid-2011: volume drifts down, holiday peak is only a small bump", fontsize=10)
save(fig, "weekly_level.png")

dow = opn.groupby(opn.date.dt.dayofweek).sales.mean()
print("\nday-of-week index (mean=1), Mon..Sun:", (dow / dow.mean()).round(2).tolist())
yr = opn.groupby(opn.date.dt.year).sales.mean()
print("mean daily units per on-sale series-day by year:", yr.round(2).to_dict(), "(composition changes as items launch)")

# ---- 5. SNAP, price ----
foods = train[train.cat_id == "FOODS"]
print(f"\nSNAP (association only, FOODS, training): mean units on SNAP days {foods[foods.snap == 1].sales.mean():.3f} "
      f"vs {foods[foods.snap == 0].sales.mean():.3f} other days; SNAP days are {foods.snap.mean():.0%} of days (CA)")
tp = train.copy()
tp["rel"] = tp.sell_price / tp.groupby("id").sell_price.transform("median")
tp["idx"] = tp.sales / tp.groupby("id").sales.transform("mean").replace(0, np.nan)
cut = pd.cut(tp.rel, [0, .85, .95, 1.05, 10], labels=["<-15%", "-15..-5%", "usual", ">+5%"])
print("price vs series median (training): share of days", cut.value_counts(normalize=True).round(3).to_dict())
print("  sales index (series mean=1) by price band:", tp.groupby(cut, observed=True).idx.mean().round(2).to_dict())
chg = panel.groupby("id").sell_price.apply(lambda s: (s.diff().abs() > 1e-6).mean())
print(f"share of series-days where price differs from previous day: mean {chg.mean():.4f} (prices are weekly)")

# ---- 6. zero runs: stock-outs / delistings / dead series (price stays listed, sales stop) ----
def zero_runs(s):
    """lengths of zero runs after the series' first sale, in open days (a trailing run counts)"""
    z = (s.to_numpy()[np.argmax(s.to_numpy() > 0):] == 0).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, z, 0]))
    return edges[1::2] - edges[::2]

first_sale = train[train.sales > 0].groupby("id").date.min()
lead = (first_sale - train.groupby("id").date.min()).dt.days
print(f"\n== zero runs, training ==\nseries whose first sale is >= 28 days after the first price: {(lead >= 28).sum()} "
      f"({lead[lead >= 28].to_dict()} days)")
runs = train.groupby("id").sales.apply(zero_runs)
rows = []
for s in ("low", "mid", "high"):
    ids = seg.index[seg == s].intersection(runs.index)
    r14 = runs[ids].apply(lambda a: a[a >= 14])
    ndays = train[train.id.isin(ids)].groupby("id").size().sum()
    rows.append((s, (r14.apply(len) > 0).sum(), r14.apply(len).sum(), int(np.median(np.concatenate(r14.values))) if r14.apply(len).sum() else 0,
                 r14.apply(sum).sum() / ndays))
print(pd.DataFrame(rows, columns=["segment", "series_with_run>=14d", "runs>=14d", "median_run_days", "share_of_days_in_such_runs"]).round(3).to_string(index=False))
last28 = opn[(opn.date > TUNING_CUTOFF - pd.Timedelta(days=28)) & (opn.date <= TUNING_CUTOFF)].groupby("id").sales.sum()
test_tot = opn[opn.date >= TEST_START].groupby("id").sales.sum()
print(f"series with zero sales in the 28 days to the cutoff: {(last28 == 0).sum()} {seg[last28[last28 == 0].index].value_counts().to_dict()}; "
      f"zero sales over the whole test window: {(test_tot == 0).sum()}")
print(f"closed days per series inside the test window: {int(panel.closed[panel.date >= TEST_START].sum() / 300)} (Dec 25 2015)")

# ---- 7. are long zero runs simultaneous across an item's three stores, or single-store? (training only) ----
W = train.pivot(index="date", columns="id", values="sales")                       # closed days already dropped (skipped, not zero)
R = pd.DataFrame(False, index=W.index, columns=W.columns)
run_rows = []                                                                      # (id, start, length) of every run >= 14 open days
for sid in W.columns:
    a = W[sid].to_numpy()
    z = (a == 0) & (np.arange(len(a)) >= np.argmax(a > 0))
    e = np.flatnonzero(np.diff(np.r_[0, z.astype(int), 0]))
    for st, en in zip(e[::2], e[1::2]):
        if en - st >= 14:
            R.iloc[st:en, R.columns.get_loc(sid)] = True
            run_rows.append((sid, st, en))
item_of = panel.groupby("id").item_id.first()
k = R.T.groupby(item_of).transform("sum").T                                        # per series-day: item's stores in a run that day
on_sale = W.notna()
p = R.to_numpy().sum() / on_sale.to_numpy().sum()
kk = k.where(R).stack()                                                            # only run-days
print(f"\n== zero runs >= 14 open days, simultaneity across the item's 3 stores (training) ==\n{len(run_rows)} runs; "
      f"{p:.1%} of on-sale series-days sit inside one")
obs = kk.value_counts(normalize=True).sort_index()
exp = pd.Series({1: (1 - p) ** 2, 2: 2 * p * (1 - p), 3: p ** 2})
print("share of run-days by number of the item's stores in a run that day (1 = this store only):")
print(pd.DataFrame({"observed": obs, "if_independent": exp}).round(3).to_string())
ov = [(k[sid].to_numpy()[st:en] > 1).mean() for sid, st, en in ((s, a, b) for s, a, b in run_rows)]
ov = pd.Series(ov)
print(f"runs with no overlap with another store's run: {(ov == 0).mean():.1%}; with >=50% of days overlapping: {(ov >= .5).mean():.1%}; "
      f"fully overlapped (every day another store also out): {(ov == 1).mean():.1%}")
seg_of = seg.reindex(R.columns)
for s in ("low", "mid", "high"):
    m = (R.loc[:, seg_of == s].to_numpy())
    kk_s = k.loc[:, seg_of == s].where(R.loc[:, seg_of == s]).stack()
    print(f"  {s}: run-days {len(kk_s)}, share with >=2 stores out {(kk_s >= 2).mean():.1%}, all 3 out {(kk_s == 3).mean():.1%}")
