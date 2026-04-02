import numpy as np


def sort_quantiles(q):
    """monotone rearrangement: fix quantile crossing by sorting each row ascending (rows = forecasts, columns = ascending alphas)"""
    return np.sort(np.asarray(q, float), axis=1)
