"""Instance-level aggregation and paired cluster bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRICS

BASE_KEYS = ["setting", "method", "kind", "group", "base_id"]


def per_instance(rows):
    if rows.empty:
        return pd.DataFrame()
    numeric = rows.copy()
    numeric[list(METRICS)] = numeric[list(METRICS)].astype(float)
    return numeric.groupby(BASE_KEYS, sort=False)[list(METRICS)].mean()


def _interval(values, draws):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan, np.nan, 0
    mean = float(values[finite].mean())
    if finite.sum() < 2:
        return mean, np.nan, np.nan, int(finite.sum())
    samples = values[draws]
    counts = np.isfinite(samples).sum(axis=1)
    means = np.divide(np.nansum(samples, axis=1), counts,
                      out=np.full(len(samples), np.nan), where=counts > 0)
    low, high = np.quantile(means[np.isfinite(means)], [0.025, 0.975])
    return mean, float(low), float(high), int(finite.sum())


def summarize(rows, *, n_base=None, seed=42, n_boot=3000):
    """Invalid answers enter valid/EF1/SD rates as zero; other metrics stay undefined."""
    base = per_instance(rows)
    if base.empty:
        return pd.DataFrame()
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot должен быть положительным целым")
    # The same draws across all variants preserve pairing, including size contrasts.
    n_base = n_base if n_base is not None else int(rows["base_id"].max()) + 1
    draws = np.random.default_rng(seed).integers(n_base, size=(n_boot, n_base))
    records = []
    for key, frame in base.groupby(level=BASE_KEYS[:-1], sort=False):
        values = frame.droplevel(BASE_KEYS[:-1]).reindex(range(n_base))
        for metric in METRICS:
            mean, low, high, defined = _interval(values[metric], draws)
            records.append(dict(zip(BASE_KEYS[:-1], key), metric=metric, mean=mean,
                                low=low, high=high, n_instances=defined,
                                expected_instances=n_base))
    return pd.DataFrame(records)


def paired_differences(rows, *, metric="ef1", n_base=None, seed=42, n_boot=3000):
    """Compute chores−goods within an instance before averaging and resampling."""
    if metric not in METRICS:
        raise ValueError(f"Неизвестная метрика: {metric}")
    base = per_instance(rows)
    if base.empty:
        return pd.DataFrame()
    n_base = n_base if n_base is not None else int(rows["base_id"].max()) + 1
    draws = np.random.default_rng(seed).integers(n_base, size=(n_boot, n_base))
    records = []
    for key, frame in base[metric].groupby(level=["setting", "method", "group"], sort=False):
        pivot = frame.droplevel(["setting", "method", "group"]).unstack("kind").reindex(range(n_base))
        if not {"goods", "chores"} <= set(pivot):
            continue
        difference = pivot["chores"] - pivot["goods"]
        mean, low, high, count = _interval(difference, draws)
        records.append(dict(zip(["setting", "method", "group"], key), metric=metric,
                            mean=mean, low=low, high=high, n_instances=count,
                            expected_instances=n_base))
    return pd.DataFrame(records)


def summary_table(rows):
    base = per_instance(rows)
    if base.empty:
        return base
    table = base.groupby(level=BASE_KEYS[:-1], sort=False).mean()
    table["n_instances"] = base.groupby(level=BASE_KEYS[:-1], sort=False).size()
    return table
