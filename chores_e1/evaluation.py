"""Approximation scores in [0, 1], with exact finite-instance benchmarks.

Goods use the paper's multiplicative fairness conventions. Chores reverse
the ratio so that one always means full satisfaction. MNW is goods-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ReferenceConfig
from .data import row_shares
from .metrics import validate_kind, validate_owners

APPROX_METRICS = ("prop_alpha", "ef_alpha", "ef1_alpha", "mms_alpha",
                  "po_alpha", "sw_ratio", "nw_ratio", "nw_ratio_lower", "efx_alpha", "prop1_alpha")
FAIRNESS_RATES = ("prop", "ef", "efx", "mms", "po", "mnw", "prop1")


def ratio(numerator, denominator):
    """Clipped satisfaction ratio: x/0 = 1, including 0/0."""
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.clip(np.divide(a, b, out=np.ones_like(a), where=b > 0), 0, 1)


@dataclass
class Evaluation:
    owners: np.ndarray | None = None
    utilities: np.ndarray | None = None
    mms: np.ndarray | None = None
    nash_welfare: np.ndarray | None = None
    best_nw: float | None = None
    nw_upper: float | None = None


def _nash_upper_bound(shares):
    """Bound indivisible MNW using positive prices and the arithmetic–geometric mean inequality."""
    n, m = shares.shape
    utility = np.full(n, 1 / n)  # equal fractional shares of each item
    upper = 1.0
    for _ in range(48):
        priced = shares / utility[:, None]
        winners = priced.argmax(axis=0)
        factor = priced[winners, np.arange(m)].sum() / n
        upper = min(upper, float(np.exp(np.log(utility).mean()) * factor))
        if factor <= 1 + 1e-6:
            break
        target = np.bincount(winners, weights=shares[winners, np.arange(m)], minlength=n)
        direction = target - utility
        lo, hi = 0.0, 1.0
        for _ in range(16):
            step = (lo + hi) / 2
            if np.sum(direction / (utility + step * direction)) > 0:
                lo = step
            else:
                hi = step
        utility += lo * direction
    return upper


def build_evaluation(weights, kind, config=None):
    """Enumerate once per instance/domain, never above the explicit budget."""
    from .reference import assignment_chunks
    validate_kind(kind)
    shares = row_shares(weights)
    n, m = shares.shape
    config = config or ReferenceConfig()
    if n ** m > config.max_allocations:
        return Evaluation(nw_upper=_nash_upper_bound(shares) if kind == "goods" else None)
    owners = np.concatenate(list(assignment_chunks(n, m, config.chunk_size)))
    mms = np.zeros(n) if kind == "goods" else np.full(n, np.inf)
    utilities = []
    for start in range(0, len(owners), config.chunk_size):
        batch = owners[start:start + config.chunk_size]
        assigned = batch[:, None, :] == np.arange(n)[None, :, None]
        bundle = np.einsum("ig,bjg->bij", shares, assigned)
        utilities.append(bundle.diagonal(axis1=1, axis2=2))
        if kind == "goods":
            mms = np.maximum(mms, bundle.min(axis=2).max(axis=0))
        else:
            mms = np.minimum(mms, bundle.max(axis=2).min(axis=0))
    utilities = np.concatenate(utilities)
    # A root makes NW comparable across different numbers of agents.
    nw = np.prod(utilities, axis=1) ** (1 / n) if kind == "goods" else None
    best_nw = float(nw.max()) if nw is not None else None
    return Evaluation(owners, utilities, mms, nw, best_nw, best_nw)


def approximation_scores(weights, owners, kind, context=None, *, atol=1e-9):
    validate_kind(kind)
    shares = row_shares(weights)
    n, m = shares.shape
    owners = validate_owners(owners, n, m)
    if len(owners) != 1:
        raise ValueError("approximation_scores принимает ровно одно распределение")
    assigned = owners[0][None, :] == np.arange(n)[:, None]
    bundle = shares @ assigned.T
    own = bundle.diagonal()
    largest = np.max(shares[:, None, :] * assigned[None, :, :], axis=2)
    # EFX means removal of ANY positively valued item (EFX+, zeros neutral).
    positive = assigned[None, :, :] & (shares[:, None, :] > 0)
    smallest = np.min(np.where(positive, shares[:, None, :], np.inf), axis=2)
    smallest = np.where(np.isfinite(smallest), smallest, 0)
    if kind == "goods":
        ef = ratio(own[:, None], bundle).min()
        ef1 = ratio(own[:, None], np.maximum(bundle - largest, 0)).min()
        efx = ratio(own[:, None], np.maximum(bundle - smallest, 0)).min()
        prop = ratio(own, 1 / n).min()
        best_extra = np.max(shares * ~assigned, axis=1)
        prop1 = ratio(own + best_extra, 1 / n).min()
        optimum = shares.max(axis=0).sum()
        sw = float(ratio(own.sum(), optimum))
    else:
        diagonal = np.eye(n, dtype=bool)
        ef = ratio(bundle, own[:, None]).min()
        ef1 = np.where(diagonal, 1, ratio(bundle, np.maximum(own - largest.diagonal(), 0)[:, None])).min()
        efx = np.where(diagonal, 1, ratio(bundle, np.maximum(own - smallest.diagonal(), 0)[:, None])).min()
        prop = ratio(1 / n, own).min()
        prop1 = ratio(1 / n, np.maximum(own - largest.diagonal(), 0)).min()
        optimum = shares.min(axis=0).sum()
        sw = float(ratio(optimum, own.sum()))
    result = dict(prop_alpha=float(prop), ef_alpha=float(ef), ef1_alpha=float(ef1),
                  efx_alpha=float(efx), prop1_alpha=float(prop1), sw_ratio=sw,
                  prop=bool(np.all(own >= 1/n - atol) if kind == "goods" else np.all(own <= 1/n + atol)),
                  ef=bool(ef >= 1 - atol), efx=bool(efx >= 1 - atol), prop1=bool(prop1 >= 1 - atol),
                  mms_alpha=None, po_alpha=None, nw_ratio=None, nw_ratio_lower=None, mms=None, po=None, mnw=None)
    context = context if context is not None else build_evaluation(weights, kind)
    if kind == "goods" and context.nw_upper:
        nw = float(np.prod(own) ** (1 / n))
        result["nw_ratio_lower"] = float(ratio(nw, context.nw_upper))
    if context.utilities is not None:
        result["mms_alpha"] = float((ratio(own, context.mms) if kind == "goods" else ratio(context.mms, own)).min())
        result["mms"] = bool(np.all(own >= context.mms - atol) if kind == "goods" else np.all(own <= context.mms + atol))
        utilities = context.utilities
        # PO-alpha measures common multiplicative improvement, whereas exact PO
        # also detects improvements to just one agent. Both are reported.
        if kind == "goods":
            improvement = np.divide(own, utilities, out=np.full_like(utilities, np.inf), where=utilities > 0)
            improvement[(utilities == 0) & (own == 0)] = 0
            po_alpha = improvement.max(axis=1).min()
            dominated = ((utilities >= own - atol).all(axis=1) & (utilities > own + atol).any(axis=1)).any()
            if context.best_nw > 0:
                nw = float(np.prod(own) ** (1/n))
                result["nw_ratio"] = float(ratio(nw, context.best_nw))
                result["mnw"] = bool(np.isclose(nw, context.best_nw, atol=atol, rtol=0))
        else:
            # Ignore zero-cost agents in common-factor improvement constraints.
            improvement = np.divide(utilities, own, out=np.full_like(utilities, np.inf), where=own > 0)
            improvement[(own == 0) & (utilities == 0)] = 0
            po_alpha = improvement.max(axis=1).min() if np.any(own > 0) else 1
            dominated = ((utilities <= own + atol).all(axis=1) & (utilities < own - atol).any(axis=1)).any()
        result["po_alpha"] = float(np.clip(po_alpha, 0, 1))
        result["po"] = bool(not dominated)
    return result
