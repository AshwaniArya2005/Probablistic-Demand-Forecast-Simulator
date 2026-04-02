"""Model interface + the three baselines of design section 7. Every model, present or future, is registered in REGISTRY and is
then covered by tests/test_models_contract.py automatically.

Interface: cls(P, seed=0, columns=None); fit(X, y=None) -> self; predict(X) -> (n,) point forecast of the P-day sum;
quantile models also predict_quantiles(X, alphas) -> (n, len(alphas)), ascending along axis 1; save(path); models.load(path).
X is a features frame; a model picks its own inputs (columns) and ignores everything else."""
import json

import numpy as np


class Model:
    name = ""
    kind = "point"          # "point" | "quantile"
    learned = False

    def __init__(self, P, seed=0, columns=None):
        self.P, self.seed = P, seed

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        raise NotImplementedError

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"name": self.name, "P": self.P, "seed": self.seed}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["P"], d["seed"])


class _Baseline(Model):
    inputs = ()             # column templates, formatted with P

    def _cols(self, X):
        cols = [c.format(P=self.P) for c in self.inputs]
        out = X[cols].to_numpy("float64")
        return out

    def predict(self, X):
        v = self._cols(X)
        if np.isnan(v[:, :1]).any():
            raise ValueError(f"{self.name}: NaN in required input {self.inputs[0].format(P=self.P)}")
        return np.maximum(self._forecast(v), 0.0)


class Naive(_Baseline):
    """the previous P days"""
    name, inputs = "naive", ("sum_last_{P}",)

    def _forecast(self, v):
        return v[:, 0]


class MA28(_Baseline):
    """28-day mean (open days) times P"""
    name, inputs = "ma28", ("mean_28",)

    def _forecast(self, v):
        return v[:, 0] * self.P


class SeasonalNaive(_Baseline):
    """same weekdays 364 days earlier; naive-P where that window is unavailable (series younger than a year)"""
    name, inputs = "snaive", ("sum_364_{P}", "sum_last_{P}")

    def _forecast(self, v):
        return np.where(np.isnan(v[:, 0]), v[:, 1], v[:, 0])

    def predict(self, X):
        v = self._cols(X)
        if np.isnan(v[:, 1]).any():
            raise ValueError("snaive: NaN in sum_last_P")
        return np.maximum(self._forecast(v), 0.0)

    def fallback_share(self, X):
        return float(np.isnan(X[f"sum_364_{self.P}"].to_numpy("float64")).mean())


REGISTRY = {c.name: c for c in (Naive, MA28, SeasonalNaive)}


def load(path):
    with open(path) as f:
        name = json.load(f)["name"]
    return REGISTRY[name].load(path)
