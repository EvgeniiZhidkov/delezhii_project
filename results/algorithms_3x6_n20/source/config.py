"""Explicit experiment settings, independent of notebook globals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: ожидается положительное целое")


@dataclass(frozen=True)
class Setting:
    n_agents: int = 3
    n_items: int = 6

    def __post_init__(self):
        positive_int(self.n_agents, "n_agents")
        positive_int(self.n_items, "n_items")

    @property
    def label(self):
        return f"{self.n_agents}×{self.n_items}"

    @property
    def allocations(self):
        return self.n_agents ** self.n_items


@dataclass(frozen=True)
class ReferenceConfig:
    method: str = "auto"  # enumerate below limit, otherwise MILP; or skip
    max_allocations: int = 100_000
    chunk_size: int = 2048
    time_limit: float = 30.0

    def __post_init__(self):
        if self.method not in {"auto", "enumerate", "milp", "skip"}:
            raise ValueError("reference.method: auto, enumerate, milp или skip")
        positive_int(self.max_allocations, "max_allocations")
        positive_int(self.chunk_size, "chunk_size")
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit должен быть положительным и конечным")


@dataclass(frozen=True)
class ModelConfig:
    label: str = "Codex"
    model: str | None = None
    reasoning_effort: str | None = None
    timeout: float = 180.0

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Задайте непустую метку модели")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model: непустое имя модели либо None")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Неизвестный reasoning_effort")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout должен быть положительным и конечным")


@dataclass(frozen=True)
class ExperimentConfig:
    settings: tuple[Setting, ...] = (Setting(),)
    n_base: int = 20
    seed: int = 20261109
    paper_lambda: int = 20
    ratios: tuple[int, ...] = (2, 5, 10)
    groups: tuple[str, ...] = ("paper", "uniform", "r=2", "r=5", "r=10")
    repeats: int = 3
    row_total: float = 100.0
    atol: float = 1e-9
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    models: tuple[ModelConfig, ...] = ()  # no model calls by default
    output_dir: str = "runs"

    def __post_init__(self):
        positive_int(self.n_base, "n_base")
        positive_int(self.repeats, "repeats")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed: неотрицательное целое")
        if type(self.paper_lambda) is not int or not 0 <= self.paper_lambda < 50:
            raise ValueError("paper_lambda: целое от 0 до 49")
        if not self.settings or any(not isinstance(s, Setting) for s in self.settings):
            raise ValueError("settings: непустой набор Setting")
        if len(set(self.settings)) != len(self.settings):
            raise ValueError("settings содержат повторы")
        if any(type(r) is not int or r <= 1 for r in self.ratios) or tuple(sorted(set(self.ratios))) != tuple(self.ratios):
            raise ValueError("ratios: разные целые > 1 по возрастанию")
        allowed = {"paper", "uniform", *(f"r={r}" for r in self.ratios)}
        if not self.groups or any(g not in allowed for g in self.groups) or len(set(self.groups)) != len(self.groups):
            raise ValueError("groups: непустой набор разных групп из paper, uniform и ratios")
        if not math.isfinite(self.row_total) or self.row_total <= 0:
            raise ValueError("row_total: положительное конечное число")
        if not math.isfinite(self.atol) or not 0 < self.atol <= 1e-6:
            raise ValueError("atol: число в (0, 1e-6]")
        labels = [m.label for m in self.models]
        if len(set(labels)) != len(labels) or set(labels) & {"Точный ориентир", "Эвристика"}:
            raise ValueError("Метки моделей должны различаться с метками алгоритмов")

    def record(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        data = dict(record)
        if "settings" in data:
            data["settings"] = tuple(Setting(**s) for s in data["settings"])
        if "reference" in data:
            data["reference"] = ReferenceConfig(**data["reference"])
        if "models" in data:
            data["models"] = tuple(ModelConfig(**m) for m in data["models"])
        for key in ("groups", "ratios"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path):
        import json
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
