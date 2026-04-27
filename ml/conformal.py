"""Split conformal calibration of upper quantiles (design section 12, Phase 7 pre-registration item 6): scale-normalised scores pooled by
velocity segment, k-th smallest with k = ceil((n+1) alpha), and a hard guard when there are too few scores."""
import math

import numpy as np

from quantiles import sort_quantiles


class InsufficientScores(ValueError):
    """fewer pooled scores than the level needs (k would exceed n); never papered over with the sample maximum"""


def n_min(alpha):
    """smallest n with ceil((n+1) alpha) <= n, i.e. n >= alpha / (1 - alpha): 4, 9, 19, 99 for 0.80, 0.90, 0.95, 0.99"""
    return math.ceil(round(alpha / (1 - alpha), 9))


def thin(n, alpha):
    """warning-only: fewer than 5 / (1 - alpha) scores makes the offset a noisy tail estimate"""
    return n < round(5 / (1 - alpha), 9)          # rounded: 5 / (1 - 0.8) is 25.000000000000004 in floating point


def scores(y, q, scale):
    """positive when the forecast was too low"""
    return (np.asarray(y, float) - np.asarray(q, float)) / np.asarray(scale, float)


def offset(score_values, alpha):
    s = np.sort(np.asarray(score_values, float))
    n = len(s)
    if n < n_min(alpha):
        raise InsufficientScores(f"alpha={alpha}: {n} scores, need at least {n_min(alpha)}")
    return float(s[math.ceil(round((n + 1) * alpha, 9)) - 1])


def segment_offsets(score_values, segments, alpha):
    """{segment: offset}, pooled over every series in the segment. Raises InsufficientScores for a segment that cannot support alpha."""
    s, sg = np.asarray(score_values, float), np.asarray(segments)
    return {g: offset(s[sg == g], alpha) for g in np.unique(sg)}


def apply_offsets(Q, segments, scale, offsets_by_alpha, alphas):
    """Q (n, len(alphas)) -> max(0, Q + offset[seg, alpha] * scale), then re-sorted across alphas. offsets_by_alpha[alpha][segment]."""
    Q = np.asarray(Q, float)
    off = np.column_stack([[offsets_by_alpha[a][g] for g in segments] for a in alphas])
    return sort_quantiles(np.maximum(Q + off * np.asarray(scale, float)[:, None], 0.0))
