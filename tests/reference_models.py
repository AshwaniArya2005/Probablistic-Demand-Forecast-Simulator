"""Test-only helpers: exact Poisson quantiles, and tiny reference learners used to validate the synthetic-truth suite itself
(and deliberately broken ones that the suite must reject). Not part of the product."""
import math

import numpy as np
import pandas as pd

from models import Model


def poisson_cdf(lam, k):
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(k) + 1))


def poisson_quantile(lam, alpha):
    """smallest integer q with CDF(q) >= alpha (the discrete quantile the design's ceil rule targets)"""
    q = 0
    while poisson_cdf(lam, q) < alpha:
        q += 1
    return q


class _Ref(Model):
    learned = True

    def __init__(self, P=7, seed=0, columns=None):
        super().__init__(P, seed)
        self.columns = list(columns)


class RefOLS(_Ref):
    name, kind, nonlinear = "ref_ols", "point", False

    def fit(self, X, y=None):
        A = np.c_[np.ones(len(X)), X[self.columns].to_numpy("float64")]
        self.coef = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)[0]
        return self

    def predict(self, X):
        return np.maximum(np.c_[np.ones(len(X)), X[self.columns].to_numpy("float64")] @ self.coef, 0.0)


class RefBinQuantile(_Ref):
    """empirical quantiles of y within bins of the first column (rounded)"""
    name, kind, nonlinear = "ref_binq", "quantile", True

    def fit(self, X, y=None):
        b = np.round(X[self.columns[0]].to_numpy("float64"))
        self.y = pd.Series(np.asarray(y, float)).groupby(b).apply(lambda s: np.sort(s.to_numpy())).to_dict()
        self.all = np.sort(np.asarray(y, float))
        return self

    def _bin(self, X):
        return [self.y.get(v, self.all) for v in np.round(X[self.columns[0]].to_numpy("float64"))]

    def predict(self, X):
        return np.array([a.mean() for a in self._bin(X)])

    def predict_quantiles(self, X, alphas):
        return np.array([np.quantile(a, alphas, method="inverted_cdf") for a in self._bin(X)]).reshape(len(X), len(alphas))


class RefZero(RefOLS):
    """broken on purpose: forecasts nothing"""
    name = "ref_zero"

    def predict(self, X):
        return np.zeros(len(X))


class RefCrossing(RefBinQuantile):
    """broken on purpose: quantiles cross"""
    name = "ref_crossing"

    def predict_quantiles(self, X, alphas):
        return super().predict_quantiles(X, alphas)[:, ::-1]
