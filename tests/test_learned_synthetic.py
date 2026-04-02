"""Synthetic data with known truth for every learned model. Linear Regression, Random Forest, XGBoost (squared error) and quantile
XGBoost join by registering in models.REGISTRY with learned=True (kind "point" or "quantile", nonlinear=True for tree models).
Until then the suite runs on two tiny reference learners, and is itself checked against deliberately broken ones, so a passing suite
means something. Thresholds are properties of the synthetic problems (noiseless, discrete-quantile slack), never project accuracy."""
import numpy as np
import pandas as pd
import pytest

from metrics import coverage_onesided
from models import REGISTRY
from reference_models import RefBinQuantile, RefCrossing, RefOLS, RefZero, poisson_cdf, poisson_quantile

ALPHAS = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
COLS = ["x1", "x2"]
registered = [c for c in REGISTRY.values() if c.learned]
POINT = [RefOLS] + [c for c in registered if c.kind == "point"]
NONLINEAR = [RefBinQuantile] + [c for c in registered if getattr(c, "nonlinear", False)]
QUANTILE = [RefBinQuantile] + [c for c in registered if c.kind == "quantile"]
ALL = [RefOLS, RefBinQuantile] + registered
name = lambda c: c.name


def uniform_frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 10, n)})


def fit_predict(cls, X, y, Xte):
    return cls(7, seed=0, columns=COLS).fit(X, y).predict(Xte)


def check_linear(cls):
    """noiseless y = 3 + 2*x1 + x2 is recovered on held-out points"""
    X, Xte = uniform_frame(4000, 1), uniform_frame(2000, 2)
    f = lambda d: 3 + 2 * d.x1 + d.x2
    p = fit_predict(cls, X, f(X), Xte)
    y = f(Xte).to_numpy()
    assert (p >= 0).all()
    assert 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() >= 0.9


def check_step(cls):
    """y = 5 above x1 = 5.5, else 0: a nonlinear model separates the two sides by most of the true gap of 5"""
    X, Xte = uniform_frame(4000, 3), uniform_frame(2000, 4)
    f = lambda d: 5.0 * (d.x1 >= 5.5)
    p = fit_predict(cls, X, f(X), Xte)
    hi = Xte.x1.to_numpy() >= 5.5
    assert p[hi].mean() - p[~hi].mean() >= 4.0


def check_zero_target(cls):
    """an all-zero target must not produce material forecasts"""
    X, Xte = uniform_frame(1000, 5), uniform_frame(500, 6)
    p = fit_predict(cls, X, np.zeros(len(X)), Xte)
    assert np.isfinite(p).all() and (p >= 0).all() and p.max() <= 0.5


def poisson_groups(n, seed):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    return pd.DataFrame({"x1": g.astype(float), "x2": rng.uniform(0, 1, n)}), rng.poisson(np.where(g == 0, 2.0, 9.0)), g


def check_quantiles(cls):
    """y | group ~ Poisson(2) or Poisson(9): predicted quantiles are near the exact ones, sorted, non-negative, and cover about alpha"""
    X, y, _ = poisson_groups(20000, 7)
    Xte, yte, g = poisson_groups(20000, 8)
    Q = cls(7, seed=0, columns=COLS).fit(X, y).predict_quantiles(Xte, ALPHAS)
    assert Q.shape == (len(Xte), len(ALPHAS)) and np.isfinite(Q).all() and (Q >= 0).all()
    assert (np.diff(Q, axis=1) >= 0).all(), "quantiles cross"
    for j, a in enumerate(ALPHAS):
        tol = 2 if a >= 0.99 else 1                              # discrete demand: neighbouring integers are equally good
        for grp, lam in ((0, 2.0), (1, 9.0)):
            assert abs(Q[g == grp, j].mean() - poisson_quantile(lam, a)) <= tol, (a, grp)
        cov = coverage_onesided(yte, Q[:, j])
        exact = 0.5 * (poisson_cdf(2.0, poisson_quantile(2.0, a)) + poisson_cdf(9.0, poisson_quantile(9.0, a)))   # coverage of the true quantiles
        assert exact >= a
        assert a - 0.04 <= cov <= exact + 0.15, (a, cov, exact)   # not under-covering; over-coverage bounded by about one integer step


@pytest.mark.parametrize("cls", POINT, ids=name)
def test_recovers_linear_truth(cls):
    check_linear(cls)


@pytest.mark.parametrize("cls", NONLINEAR, ids=name)
def test_learns_a_step(cls):
    check_step(cls)


@pytest.mark.parametrize("cls", ALL, ids=name)
def test_all_zero_target(cls):
    check_zero_target(cls)


@pytest.mark.parametrize("cls", QUANTILE, ids=name)
def test_quantiles_near_truth_covered_and_sorted(cls):
    check_quantiles(cls)


def test_the_suite_rejects_broken_models():
    with pytest.raises(AssertionError):
        check_linear(RefZero)                    # forecasts nothing
    with pytest.raises(AssertionError):
        check_quantiles(RefCrossing)             # crossing quantiles
    with pytest.raises(AssertionError):
        check_step(RefOLS)                       # a linear model cannot reproduce a hard step to within 20% of its height
