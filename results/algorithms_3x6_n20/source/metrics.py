"""Independent cardinal EF1 and ordinal SD checks for additive goods/chores.

SD counts items at every *distinct positive* weight threshold. Equal weights
stay tied and zero-valued items are indifferent to the null item. Removing
the highest-weight item is simultaneously optimal at every threshold.
"""
from __future__ import annotations

import numpy as np

from .data import row_shares, validate_weights

METRICS = ("valid", "ef1", "d", "envy", "total", "gap", "sd_ef", "sd_ef1",
           "sd_envy", "sd_d", "sd_ef_pairs", "sd_ef1_pairs",
           "wrong_side_only", "goods_rule_only")


def validate_kind(kind):
    if kind not in {"goods", "chores"}:
        raise ValueError("kind: goods или chores")


def validate_owners(owners, n, m):
    owners = np.asarray(owners)
    if owners.ndim == 1:
        owners = owners[None, :]
    if owners.ndim != 2 or owners.shape[1] != m or len(owners) == 0:
        raise ValueError("Назначение должно иметь размер m или batch×m")
    if owners.dtype.kind not in "iu" or (owners < 0).any() or (owners >= n).any():
        raise ValueError("Получатели: целые ID от 0 до n−1")
    return owners


def allocation_scores(weights, owners, kind, *, row_total=100.0, atol=1e-9):
    validate_kind(kind)
    shares = row_shares(weights)
    n, m = shares.shape
    owners = validate_owners(owners, n, m)
    assigned = owners[:, None, :] == np.arange(n)[None, :, None]
    bundle = np.einsum("ig,bjg->bij", shares, assigned)
    own = bundle.diagonal(axis1=1, axis2=2)
    # Avoid the old batch×n×n×m temporary: use one evaluator at a time.
    largest = np.stack([np.max(shares[i] * assigned, axis=-1) for i in range(n)], axis=1)
    goods_envy = bundle - own[:, :, None]
    if kind == "goods":
        envy, residual = goods_envy, goods_envy - largest
    else:
        envy = -goods_envy
        residual = envy - largest.diagonal(axis1=1, axis2=2)[:, :, None]
    cheapest = np.stack([np.min(np.where(assigned, shares[i], np.inf), axis=-1)
                         for i in range(n)], axis=1)
    cheapest = np.where(assigned.any(axis=-1)[:, None, :], cheapest, 0)
    diagonal = np.eye(n, dtype=bool)
    return {
        "d": np.maximum(np.where(diagonal, 0, residual).max(axis=(1, 2)), 0),
        "envy": np.maximum(np.where(diagonal, 0, envy).max(axis=(1, 2)), 0),
        "total": row_total * own.sum(axis=1),
        "wrong_side_pass": np.where(diagonal, 0, -goods_envy + cheapest).max(axis=(1, 2)) <= atol,
        "goods_rule_pass": np.where(diagonal, 0, goods_envy - largest).max(axis=(1, 2)) <= atol,
    }


def sd_scores(weights, owners, kind):
    """SD-EF/SD-EF1, worst prefix deficit / m, and fraction of passing pairs.

    For goods: #own(weight>=q) >= #other(weight>=q).
    For chores: #own(cost>=q) <= #other(cost>=q).
    No arbitrary tie-breaking; thresholds q=0 are excluded in both domains.
    """
    validate_kind(kind)
    weights = validate_weights(weights)
    n, m = weights.shape
    owners = validate_owners(owners, n, m)
    assigned = owners[:, None, :] == np.arange(n)[None, :, None]
    deficits, residuals = [], []
    for i in range(n):
        thresholds = np.unique(weights[i][weights[i] > 0])
        prefixes = weights[i][None, :] >= thresholds[:, None]
        counts = np.einsum("kg,bjg->bjk", prefixes.astype(int), assigned.astype(int))
        for j in range(n):
            if i == j:
                continue
            delta = counts[:, j] - counts[:, i] if kind == "goods" else counts[:, i] - counts[:, j]
            removable = counts[:, j] if kind == "goods" else counts[:, i]
            deficits.append(np.maximum(delta.max(axis=1, initial=0), 0))
            residuals.append(np.maximum((delta - (removable > 0)).max(axis=1, initial=0), 0))
    if not deficits:  # one participant: all pairwise conditions are vacuous
        zeros = np.zeros(len(owners))
        ones = np.ones(len(owners))
        return {"sd_ef": ones.astype(bool), "sd_ef1": ones.astype(bool),
                "sd_envy": zeros, "sd_d": zeros, "sd_ef_pairs": ones, "sd_ef1_pairs": ones}
    deficit, residual = np.stack(deficits, axis=1), np.stack(residuals, axis=1)
    return {"sd_ef": (deficit == 0).all(axis=1), "sd_ef1": (residual == 0).all(axis=1),
            "sd_envy": deficit.max(axis=1) / m, "sd_d": residual.max(axis=1) / m,
            "sd_ef_pairs": (deficit == 0).mean(axis=1),
            "sd_ef1_pairs": (residual == 0).mean(axis=1)}


def answer_metrics(weights, owners, kind, reference=None, *, row_total=100.0, atol=1e-9):
    validate_kind(kind)
    if owners is None:
        return {name: (False if name in {"valid", "ef1", "sd_ef", "sd_ef1"} else None)
                for name in METRICS}
    scores = allocation_scores(weights, owners, kind, row_total=row_total, atol=atol)
    if len(scores["d"]) != 1:
        raise ValueError("answer_metrics принимает ровно одно распределение")
    d, total = float(scores["d"][0]), float(scores["total"][0])
    ef1 = d <= atol
    gap = None
    if reference is not None and reference.certified and abs(d - reference.best_d) <= atol:
        difference = reference.best_total - total if kind == "goods" else total - reference.best_total
        gap = max(0.0, difference)
    sd = {key: value[0].item() for key, value in sd_scores(weights, owners, kind).items()}
    return {"valid": True, "ef1": ef1, "d": d, "envy": float(scores["envy"][0]),
            "total": total, "gap": gap, **sd,
            "wrong_side_only": bool(scores["wrong_side_pass"][0] and not ef1) if kind == "chores" else None,
            "goods_rule_only": bool(scores["goods_rule_pass"][0] and not ef1) if kind == "chores" else None}
