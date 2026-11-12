"""Reproducible paired inputs. Adding a size/group does not change old inputs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ExperimentConfig, Setting


@dataclass(frozen=True)
class Instance:
    setting: Setting
    base_id: int
    group: str
    weights: np.ndarray

    @property
    def key(self):
        return self.setting.label, self.base_id, self.group

    def record(self):
        return {"setting": self.setting.label, "base_id": self.base_id,
                "group": self.group, "weights": self.weights.tolist()}


def validate_weights(weights):
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or min(values.shape, default=0) < 1:
        raise ValueError("Ожидается непустая матрица оценок n×m")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Оценки должны быть конечными и неотрицательными")
    totals = values.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Для нормировки нужна положительная конечная сумма каждой строки")
    return values


def row_shares(weights):
    values = validate_weights(weights)
    return values / values.sum(axis=1, keepdims=True)


def generate_instances(config: ExperimentConfig):
    instances = []
    for setting in config.settings:
        n, m = setting.n_agents, setting.n_items
        for base_id in range(config.n_base):
            # Preserve the original 3×6 pilot bit-for-bit; other sizes have independent streams.
            suffix = [] if (n, m) == (3, 6) else [n, m]
            article, uniform, binary = [np.random.default_rng(
                np.random.SeedSequence([config.seed, base_id, stream, *suffix])
            ) for stream in range(3)]
            p = article.dirichlet(np.ones(m))
            totals = article.integers((50 - config.paper_lambda) * m,
                                     (50 + config.paper_lambda) * m + 1, size=n)
            paper = np.stack([article.multinomial(int(total), p) for total in totals])
            independent = uniform.integers(1, 11, size=(n, m))
            expensive = binary.random((n, m)) < 0.5
            values = {"paper": paper, "uniform": independent}
            values.update({f"r={r}": np.where(expensive, r, 1) for r in config.ratios})
            for group in config.groups:
                weights = values[group]
                weights.setflags(write=False)
                instances.append(Instance(setting, base_id, group, weights))
    return instances
