"""Learned point models of Phase 6: Linear Regression, Random Forest, XGBoost (squared error). Rules are pre-registered in
docs/design.md section 12. All share: a model picks its inputs (`columns`, default features.columns(P), minus `drop` for ablations);
optional target normalisation (train on Y / scale, predict yhat * scale, scale = max(mean_28, floor) * P); predictions clipped at 0.
Ids: one-hot for Linear Regression, arbitrary ordinal codes for the trees (unseen levels: all-zero one-hot / code -1)."""
import base64
import copy
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import features
from models import REGISTRY, Model
from quantiles import sort_quantiles

EARLY_STOP_DAYS = 56


def early_stop_split(dates, P, days=EARLY_STOP_DAYS):
    """temporal holdout inside a fit set, with an embargo. With L the last fit origin: validation = origins after L - days;
    the early-stopping stage trains on origins whose target window ended by L - days, so no training target day is a validation target day."""
    d = pd.DatetimeIndex(dates)
    edge = d.max() - pd.Timedelta(days=days)
    return np.flatnonzero((d + pd.Timedelta(days=P)) <= edge), np.flatnonzero(d > edge)


class Learned(Model):
    learned, kind, nonlinear = True, "point", True
    defaults = {}

    def __init__(self, P, seed=0, columns=None, normalize=False, floor=1 / 28, drop=(), **params):
        super().__init__(P, seed)
        cols = list(columns) if columns is not None else features.columns(P)
        self.columns = [c for c in cols if c not in set(drop)]
        self.normalize, self.floor, self.drop = normalize, floor, tuple(drop)
        self.params = {**self.defaults, **params}

    # -- shared plumbing --
    def scale(self, X):
        return np.fmax(X["mean_28"].to_numpy("float64"), self.floor) * self.P

    def _split(self, X):
        ids = [c for c in self.columns if c in features.STATIC]
        return [c for c in self.columns if c not in ids], ids

    def _codes(self, X, fit):
        num, ids = self._split(X)
        if fit:
            self.maps = {c: {v: i for i, v in enumerate(sorted(X[c].astype(str).unique()))} for c in ids}
        C = np.column_stack([X[c].astype(str).map(self.maps[c]).fillna(-1).to_numpy("float32") for c in ids]) if ids else np.zeros((len(X), 0), "float32")
        return X[num].to_numpy("float32"), C

    def fit(self, X, y=None):
        y = np.asarray(y, "float64")
        if self.normalize:
            y = y / self.scale(X)
        self._fit(X, y)
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0)
        p = np.asarray(self._predict(X), "float64")
        if self.normalize:
            p = p * self.scale(X)
        return np.maximum(p, 0.0)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"name": self.name, "P": self.P, "seed": self.seed, "state": base64.b64encode(pickle.dumps(self)).decode()}, f)

    @classmethod
    def load(cls, path):
        # pickle: only ever load files written by save() in this repo (local model artifacts, git-ignored); never untrusted input
        with open(path) as f:
            return pickle.loads(base64.b64decode(json.load(f)["state"]))


class LR(Learned):
    """standardised numerics (fit-set mean/std), median-imputed with a missing-indicator where the fit set had NaN, one-hot ids, OLS"""
    name, nonlinear = "lr", False

    def _design(self, X, fit):
        N, C = self._codes(X, fit)
        N = N.astype("float64")
        if fit:
            self.med = np.array([np.nanmedian(N[:, j]) if (~np.isnan(N[:, j])).any() else 0.0 for j in range(N.shape[1])])
            self.miss = np.flatnonzero(np.isnan(N).any(0))
        flags = np.isnan(N[:, self.miss]).astype("float64")
        N = np.where(np.isnan(N), self.med, N)
        if fit:
            self.mu, self.sd = N.mean(0), N.std(0)
            self.sd[self.sd == 0] = 1.0
        N = (N - self.mu) / self.sd
        onehot = [(C[:, j : j + 1] == np.arange(len(m))).astype("float64") for j, m in enumerate(self.maps.values())]
        return np.hstack([N, flags, *onehot])

    def _fit(self, X, y):
        self.model = LinearRegression().fit(self._design(X, True), y)

    def _predict(self, X):
        return self.model.predict(self._design(X, False))


class _Trees(Learned):
    def _matrix(self, X, fit):
        N, C = self._codes(X, fit)
        return np.hstack([N, C])


class RF(_Trees):
    """100 trees, max_features 0.5; training size bounded by max_samples rows per tree (20,000) and the depth cap"""
    name = "rf"
    defaults = dict(max_depth=16, min_samples_leaf=20, n_estimators=100, max_features=0.5, max_samples=20000)

    def _fit(self, X, y):
        p, M = self.params, self._matrix(X, True)
        self.model = RandomForestRegressor(n_estimators=p["n_estimators"], max_features=p["max_features"], bootstrap=True,
                                           max_samples=min(p["max_samples"], len(M)), max_depth=p["max_depth"],
                                           min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=self.seed).fit(M, y)
        self.model.set_params(n_jobs=1)   # parallel predict sums trees in thread-completion order: not bit-reproducible

    def _predict(self, X):
        return self.model.predict(self._matrix(X, False))


class XGB(_Trees):
    """squared error, hist. With a `date` column: early stopping on a temporal holdout with an embargo, then refit on the whole fit set
    with best_iteration + 1 rounds. Without dates (synthetic tests): a fixed round count."""
    name = "xgb"
    defaults = dict(max_depth=6, min_child_weight=30, learning_rate=0.05, max_rounds=1000, patience=50, fixed_rounds=300)

    def _param(self):
        p = self.params
        return dict(objective="reg:squarederror", tree_method="hist", learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                    min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, seed=self.seed,
                    **({"nthread": p["nthread"]} if p.get("nthread") else {}))         # fixed thread count keeps runs reproducible

    def _fit(self, X, y):
        p, M, param = self.params, self._matrix(X, True), self._param()
        rounds = p["fixed_rounds"]
        self.hit_cap_ = False
        if "date" in X.columns:
            tr, va = early_stop_split(X["date"], self.P)
            if len(tr) and len(va):
                bst = xgb.train(param, xgb.DMatrix(M[tr], label=y[tr]), p["max_rounds"], evals=[(xgb.DMatrix(M[va], label=y[va]), "val")],
                                early_stopping_rounds=p["patience"], verbose_eval=False)
                rounds = bst.best_iteration + 1
                self.hit_cap_ = rounds >= p["max_rounds"]          # reported: the pre-registered round cap bound
        self.n_rounds_ = rounds
        self.model = xgb.train(param, xgb.DMatrix(M, label=y), rounds)

    def _predict(self, X):
        return self.model.predict(xgb.DMatrix(self._matrix(X, False)))


class XGBQ(XGB):
    """XGBoost quantile regression (`reg:quantileerror`), one joint multi-quantile model per horizon for ALPHAS. Output: scaled back if
    normalised, clipped at 0, sorted along the quantile axis (monotone rearrangement). `predict` is the median.
    Without dates a fixed 2,000 rounds are used (the 0.99 quantile converges slowly)."""
    name, kind = "xgb_q", "quantile"
    ALPHAS = (0.10, 0.50, 0.80, 0.90, 0.95, 0.99)
    defaults = dict(XGB.defaults, fixed_rounds=2000)

    def _param(self):
        return dict(super()._param(), objective="reg:quantileerror", quantile_alpha=np.array(self.ALPHAS))

    def _raw(self, X):
        """the model's reported quantiles before clipping: XGBoost's joint `predict` returns the per-target tree sums already SORTED"""
        q = np.asarray(self.model.predict(xgb.DMatrix(self._matrix(X, False))), "float64").reshape(len(X), len(self.ALPHAS))
        return q * self.scale(X)[:, None] if self.normalize else q

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_singles"}      # the sliced boosters are rebuilt on demand

    def _single_boosters(self):
        """one single-target booster per quantile, sliced from the joint model's JSON (only that target's trees, its own base score). XGBoost 3.4.1
        has two defects here: joint `pred_contribs` does not reconcile with joint `predict` (it explains the unsorted tree sums, predict returns them
        sorted), and the `shap` package inherits it. A sliced booster is exact: its prediction is the target's tree sum and its contributions add up to it."""
        if getattr(self, "_singles", None) is None:
            raw = json.loads(bytes(self.model.save_raw("json")))
            alphas, base = self.ALPHAS, json.loads(raw["learner"]["learner_model_param"]["base_score"].replace("E", "e"))
            out = []
            for t in range(len(alphas)):
                r = copy.deepcopy(raw)
                m = r["learner"]["gradient_booster"]["model"]
                trees = [copy.deepcopy(m["trees"][i]) for i, g in enumerate(m["tree_info"]) if g == t]
                for j, tr in enumerate(trees):
                    tr["id"] = j
                m["trees"], m["tree_info"] = trees, [0] * len(trees)
                m["gbtree_model_param"]["num_trees"] = str(len(trees))
                m["iteration_indptr"] = list(range(len(trees) + 1))
                r["learner"]["learner_model_param"]["base_score"] = "[%s]" % repr(float(base[t]))
                r["learner"]["learner_model_param"]["num_target"] = "1"
                r["learner"]["objective"]["quantile_loss_param"]["quantile_alpha"] = str(alphas[t])
                out.append(xgb.Booster(model_file=bytearray(json.dumps(r).encode())))
            self._singles = out
        return self._singles

    def _unsorted(self, X):
        """per-target tree sums before XGBoost's internal sorting (units)"""
        d = xgb.DMatrix(self._matrix(X, False))
        q = np.column_stack([b.predict(d) for b in self._single_boosters()]).astype("float64")
        return q * self.scale(X)[:, None] if self.normalize else q

    def crossing_share(self, X):
        """share of rows whose per-target tree sums cross, i.e. rows XGBoost's internal sort changed (diagnostic; measured on the unsorted sums)"""
        return float((np.diff(self._unsorted(X), axis=1) < 0).any(axis=1).mean()) if len(X) else 0.0

    def feature_names(self, X=None):
        num, ids = self._split(X)
        return num + ids

    def contributions(self, X, alphas):
        """exact per-quantile TreeSHAP contributions of the REPORTED (sorted) quantiles, in units. For each row and rank r the contributions come
        from whichever target's trees produced the r-th smallest tree sum, so they add up to the reported value even where quantiles cross.
        Returns dict: phi (n, len(alphas), F) feature contributions, bias (n, len(alphas)) baseline, value (n, len(alphas)) = bias + sum(phi) = the
        reported quantile before clipping at 0, names, source (n, len(alphas)) the target index that supplied each value."""
        idx = [self.ALPHAS.index(a) for a in alphas]
        d = xgb.DMatrix(self._matrix(X, False))
        parts = np.stack([b.predict(d, pred_contribs=True) for b in self._single_boosters()])          # (6, n, F+1), model space
        order = np.argsort(parts.sum(-1).T, axis=1, kind="stable")                                       # order[:, r]: target giving the r-th smallest
        src = order[:, idx]
        phi = parts[src, np.arange(len(X))[:, None], :]                                                  # (n, k, F+1)
        if self.normalize:
            phi = phi * self.scale(X)[:, None, None]                                                     # ratio space -> units
        return dict(phi=phi[:, :, :-1], bias=phi[:, :, -1], value=phi.sum(-1), names=self.feature_names(X), source=src)

    def predict_quantiles(self, X, alphas):
        idx = [self.ALPHAS.index(a) for a in alphas]                  # ValueError if an alpha was not trained
        if len(X) == 0:
            return np.zeros((0, len(idx)))
        return sort_quantiles(np.maximum(self._raw(X), 0.0))[:, idx]

    def predict(self, X):
        return self.predict_quantiles(X, [0.50])[:, 0]


REGISTRY.update({c.name: c for c in (LR, RF, XGB, XGBQ)})
