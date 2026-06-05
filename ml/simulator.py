"""Phase 10 engine: periodic-review, order-up-to inventory replay with lost sales, exactly as defined in docs/design.md section 9 and the Phase 10 definitions of
section 12. Pure numpy on integer arrays; no data, no model. The policies (order-up-to levels S) come from ml/sim_policies.py.

Timing: the order is placed after close of review day t and arrives before opening of day t + L + 1. The cycle of review t is the seven days t + L + 1 .. t + L + R,
the arrival window of that order. Everything is integer. Demand is replayed, never simulated; unmet demand is lost."""
from dataclasses import dataclass

import numpy as np

REVIEW = 7
RHOS = (4, 9, 19, 99)
HOLD_YEAR, COST_SHARE = 0.25, 0.70                # holding 25% of unit cost per year; unit cost 70% of price


@dataclass
class Replay:
    onhand: np.ndarray        # (T, N) end-of-day on-hand (int64)
    lost: np.ndarray          # (T, N) unmet demand per day
    served: np.ndarray        # (T, N)
    orders: np.ndarray        # (K, N) order quantity placed at each review after the first (row 0 is zeros)


def replay(demand, reviews, S, L, R=REVIEW, lead=None):
    """demand (T, N) non-negative ints indexed by day; reviews: K increasing day indices exactly R apart; S (K, N) order-up-to levels (ints >= 0).
    Starts after the close of reviews[0] with on-hand = S[0] and nothing on order. Days before reviews[0] + 1 are left as zeros. Simulates up to
    reviews[-1] + L + R, the last day of the last cycle. lead: optional (K, N) integer lead time of the order placed at each review (random-lead sensitivity; the policies still plan
    with the nominal L and the cycles keep their nominal windows; orders may cross); None means every order takes L."""
    if not (np.issubdtype(np.asarray(S).dtype, np.integer) and np.issubdtype(np.asarray(demand).dtype, np.integer)):
        raise TypeError("S and demand must be integer arrays (units are indivisible; policies round S up with ceil)")
    demand, S, reviews = np.asarray(demand, "int64"), np.asarray(S, "int64"), np.asarray(reviews, "int64")
    K, N = S.shape
    if (np.diff(reviews) != R).any() or (S < 0).any() or (demand < 0).any() or L < 0:
        raise ValueError("reviews must be R days apart; S, demand and L must be non-negative")
    end = int(reviews[-1]) + L + R
    if demand.shape != (demand.shape[0], N) or demand.shape[0] <= end:
        raise ValueError(f"demand must have {N} columns and cover day {end}")
    on = S[0].copy()
    if lead is not None:
        lead = np.asarray(lead, "int64")
        if lead.shape != (K, N) or (lead < 0).any():
            raise ValueError("lead must be a non-negative (K, N) integer array")
    arrivals = np.zeros((end + (L if lead is None else int(lead.max())) + 2, N), "int64")
    onhand, lost, served = (np.zeros((end + 1, N), "int64") for _ in range(3))
    orders = np.zeros((K, N), "int64")
    rev_at = {int(d): k for k, d in enumerate(reviews)}
    for d in range(int(reviews[0]) + 1, end + 1):
        on += arrivals[d]
        s = np.minimum(on, demand[d])
        served[d], lost[d], onhand[d] = s, demand[d] - s, on - s
        on = on - s
        k = rev_at.get(d)
        if k is not None and k > 0:
            q = np.maximum(S[k] - (on + arrivals[d + 1:].sum(0)), 0)          # order-up-to: S minus on-hand minus on-order
            orders[k] = q
            if lead is None:
                arrivals[d + L + 1] += q
            else:
                arrivals[d + lead[k] + 1, np.arange(N)] += q
    return Replay(onhand, lost, served, orders)


def cycle_days(reviews, L, R=REVIEW):
    """(K, R) day indices of each review's cycle"""
    return np.asarray(reviews, "int64")[:, None] + L + 1 + np.arange(R)


def cycle_table(rep, demand, reviews, L, metric_k, R=REVIEW, P=None):
    """per metric cycle and series: units demanded, lost, and on-hand summed over the cycle's days. metric_k = indices into `reviews` that count
    (warm-up reviews are left out by the caller). With P given, "zero" flags the cells whose demand over the protection interval t + 1 .. t + P was zero (the hindsight
    diagnostic of design.md section 9). Returns dict of (m, N) arrays."""
    days = cycle_days(reviews, L, R)[np.asarray(metric_k)]
    dem = np.asarray(demand, "int64")[days]                    # (m, R, N)
    lost = rep.lost[days]
    out = dict(demanded=dem.sum(1), lost=lost.sum(1), onhand=rep.onhand[days].sum(1), stockout=(lost.sum(1) > 0).astype("int64"))
    if P is not None:
        window = np.asarray(reviews, "int64")[np.asarray(metric_k)][:, None] + 1 + np.arange(P)
        out["zero"] = (np.asarray(demand, "int64")[window].sum(1) == 0).astype("int64")
    return out


def cost_parts(price, onhand_days, lost, rho):
    """holding cost, stockout cost per series: holding = 0.7 * price * 0.25 / 365 per unit-day; stockout = rho x the holding cost of one unit over R = 7 days per lost unit"""
    h = COST_SHARE * np.asarray(price, float) * HOLD_YEAR / 365.0
    return h * onhand_days, rho * h * REVIEW * lost


def matched_inventory(ref_fill, fill, inv):
    """comparator's inventory at the fill rate `ref_fill`, linear interpolation on the comparator's curve sorted by fill rate; NaN outside its range (no
    extrapolation). fill, inv: (..., m) with the same leading shape as ref_fill (...,). Vectorised over the leading axes (bootstrap resamples)."""
    fill, inv, ref = np.asarray(fill, float), np.asarray(inv, float), np.asarray(ref_fill, float)
    o = np.argsort(fill, axis=-1, kind="stable")
    f, v = np.take_along_axis(fill, o, -1), np.take_along_axis(inv, o, -1)
    m = f.shape[-1]
    pos = (f < ref[..., None]).sum(-1)                          # number of points strictly below ref
    lo, hi = np.clip(pos - 1, 0, m - 1), np.clip(pos, 0, m - 1)
    f0, f1 = np.take_along_axis(f, lo[..., None], -1)[..., 0], np.take_along_axis(f, hi[..., None], -1)[..., 0]
    v0, v1 = np.take_along_axis(v, lo[..., None], -1)[..., 0], np.take_along_axis(v, hi[..., None], -1)[..., 0]
    w = np.where(f1 > f0, (ref - f0) / np.where(f1 > f0, f1 - f0, 1.0), 1.0)
    out = v0 + w * (v1 - v0)
    ok = (ref >= f[..., 0]) & (ref <= f[..., -1])
    return np.where(ok, out, np.nan)


def matched_reduction(fill_q, inv_q, fill_c, inv_c, anchor):
    """inventory reduction of the headline policy (q) against a comparator (c) at matched fill rate, as 1 - inventory(q) / inventory(c). fill_*, inv_*: (..., m) curves.
    anchor 'q' (the pre-registered statistic): at each of q's points, the comparator's inventory interpolated at q's fill rate; NaN where the comparator's curve does not reach.
    anchor 'c' (added before the test-window run, see design.md): at each of the comparator's points, q's inventory interpolated at the comparator's fill rate; NaN where q's curve
    does not reach. Either way there is no extrapolation. Returns (..., m) matching the anchor's curve."""
    fill_q, inv_q, fill_c, inv_c = (np.asarray(a, float) for a in (fill_q, inv_q, fill_c, inv_c))
    if anchor == "q":
        return 1 - inv_q / np.stack([matched_inventory(fill_q[..., j], fill_c, inv_c) for j in range(fill_q.shape[-1])], axis=-1)
    if anchor == "c":
        return 1 - np.stack([matched_inventory(fill_c[..., j], fill_q, inv_q) for j in range(fill_c.shape[-1])], axis=-1) / inv_c
    raise ValueError("anchor must be 'q' or 'c'")


def item_sums(table, ids, items, cycle_mask=None, cell_mask=None):
    """aggregate per-(cycle, series) arrays to per-item sums: dict of (n_items,) arrays for demanded, lost, onhand, stockout, cycles, days.
    items: the sorted unique item ids; cycle_mask (m,) selects cycles (sub-period); cell_mask (m, N) selects (cycle, series) cells (the zero-demand diagnostic drops cells)."""
    m = table["lost"].shape[0]
    keep = np.ones((m, len(ids)), bool)
    if cycle_mask is not None:
        keep &= np.asarray(cycle_mask, bool)[:, None]
    if cell_mask is not None:
        keep &= np.asarray(cell_mask, bool)
    idx = {it: i for i, it in enumerate(items)}
    col = np.array([idx[i] for i in ids])
    out = {k: np.bincount(col, weights=(table[k] * keep).sum(0), minlength=len(items)) for k in ("demanded", "lost", "onhand", "stockout")}
    out["cycles"] = np.bincount(col, weights=keep.sum(0), minlength=len(items))
    out["days"] = out["cycles"] * REVIEW
    return out


def curve_stats(W, sums):
    """weights W (B, n_items) item counts of resamples -> dict of (B,) fill rate, cycle service, average on-hand per series-day for one policy setting"""
    dem, lost, oh, so, cyc, days = (W @ sums[k] for k in ("demanded", "lost", "onhand", "stockout", "cycles", "days"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(fill=1 - lost / dem, cycle_service=1 - so / cyc, inv=oh / days)


def bootstrap_weights(n_items, B=10000, seed=0):
    """item counts of B cluster-bootstrap resamples over the items (an item's three store-series stay together): (B, n_items), rows sum to n_items"""
    return np.random.default_rng(seed).multinomial(n_items, np.full(n_items, 1.0 / n_items), size=B).astype(float)
