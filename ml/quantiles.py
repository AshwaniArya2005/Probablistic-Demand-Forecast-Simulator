from statistics import NormalDist

import numpy as np


def sort_quantiles(q):
    """monotone rearrangement: fix quantile crossing by sorting each row ascending (rows = forecasts, columns = ascending alphas)"""
    return np.sort(np.asarray(q, float), axis=1)


def normal_z(alpha):
    return NormalDist().inv_cdf(alpha)


def pooled_sigma(residual_over_scale, segments):
    """RMSE of scale-normalised residuals pooled by velocity segment: {segment: sigma}"""
    r, sg = np.asarray(residual_over_scale, float), np.asarray(segments)
    return {s: float(np.sqrt((r[sg == s] ** 2).mean())) for s in np.unique(sg)}


def point_policy_quantile(yhat, sigma_seg, scale, alpha):
    """design section 8: S = yhat + z_alpha * sigma_seg * scale (sigma_seg pooled, scaled back by the series' own scale), clipped at 0"""
    return np.maximum(np.asarray(yhat, float) + normal_z(alpha) * np.asarray(sigma_seg, float) * np.asarray(scale, float), 0.0)
