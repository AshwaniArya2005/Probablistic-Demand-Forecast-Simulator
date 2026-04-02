"""Metrics of design section 6. Arrays are 1-d numpy; `ids` labels the series of each row."""
import numpy as np
import pandas as pd


def wape(y, yhat):
    """pooled: sum|y - yhat| / sum y"""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if y.sum() <= 0:
        raise ValueError("WAPE undefined: actuals sum to zero")
    return np.abs(y - yhat).sum() / y.sum()


def mae(y, yhat):
    return float(np.abs(np.asarray(y, float) - np.asarray(yhat, float)).mean())


def rmse(y, yhat):
    return float(np.sqrt(((np.asarray(y, float) - np.asarray(yhat, float)) ** 2).mean()))


def mape_nonzero(y, yhat):
    """reference only (never used for selection): breaks at zero, so zero actuals are dropped"""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y > 0
    return float((np.abs(y[m] - yhat[m]) / y[m]).mean())


def naive_scale(feats, P, before):
    """per-series in-sample MAE of the naive-P forecast (y_t - y_{t-P}) on origins whose target ended on or before `before`.
    naive-P at origin t is sum_last_P; both columns exist in the features table."""
    f = feats[feats.date + pd.Timedelta(days=P) <= before]
    return (f[f"y_p{P}"] - f[f"sum_last_{P}"]).abs().groupby(f.id).mean()


def _per_series(err, ids):
    return pd.Series(np.asarray(err, float)).groupby(np.asarray(ids)).mean()


def mase(y, yhat, ids, scale):
    """mean over series of (series MAE / scale). Series with zero or missing scale are excluded and counted.
    returns (value, n_series_used, n_excluded)"""
    mae_s = _per_series(np.abs(np.asarray(y, float) - np.asarray(yhat, float)), ids)
    sc = scale.reindex(mae_s.index)
    ok = sc.notna() & (sc > 0)
    return float((mae_s[ok] / sc[ok]).mean()), int(ok.sum()), int((~ok).sum())


def pinball(y, q, alpha):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.maximum(alpha * d, (alpha - 1) * d).mean())


def scaled_pinball(y, q, alpha, ids, scale):
    """per-series mean pinball / the same per-series scale as MASE, averaged over series (zero-scale series excluded)"""
    d = np.asarray(y, float) - np.asarray(q, float)
    pb = _per_series(np.maximum(alpha * d, (alpha - 1) * d), ids)
    sc = scale.reindex(pb.index)
    ok = sc.notna() & (sc > 0)
    return float((pb[ok] / sc[ok]).mean()), int(ok.sum()), int((~ok).sum())


def coverage_interval(y, lo, hi):
    y = np.asarray(y, float)
    return float(((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))).mean())


def coverage_onesided(y, q):
    """share of actuals at or below the predicted quantile (target: alpha, or more for discrete demand)"""
    return float((np.asarray(y, float) <= np.asarray(q, float)).mean())
