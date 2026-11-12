"""Exhaustive reference and simple heuristics for small E1 cases."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import numpy as np

from .config import ReferenceConfig
from .data import row_shares, validate_weights
from .metrics import allocation_scores, validate_kind


@dataclass
class ReferenceResult:
    method: str
    status: str
    owners: list[int] | None = None
    best_d: float | None = None
    best_total: float | None = None
    certified: bool = False
    seconds: float = 0.0
    message: str = ""

    def record(self):
        return asdict(self)


def assignment_chunks(n, m, chunk_size=2048):
    """Lexicographic batches of assignments without materializing n**m rows."""
    total = n ** m
    if total > np.iinfo(np.int64).max:
        raise ValueError("Слишком много назначений для индексируемого перебора")
    for start in range(0, total, chunk_size):
        ids = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        batch = np.empty((len(ids), m), dtype=np.int64)
        for g in range(m - 1, -1, -1):
            batch[:, g] = ids % n
            ids //= n
        yield batch


def exact_reference(weights, kind, config=None, *, row_total=100.0, atol=1e-9):
    config = config or ReferenceConfig()
    validate_kind(kind)
    weights = validate_weights(weights)
    n, m = weights.shape
    total = n ** m
    if total > config.max_allocations:
        return ReferenceResult("enumerate", "skipped", message=f"{total} > max_allocations={config.max_allocations}")
    started = time.perf_counter()
    # Store only two scalars per assignment. Peak bundle memory depends on chunk_size.
    all_d, all_total = np.empty(total), np.empty(total)
    start = 0
    for batch in assignment_chunks(n, m, config.chunk_size):
        scores = allocation_scores(weights, batch, kind, row_total=row_total, atol=atol)
        stop = start + len(batch)
        all_d[start:stop], all_total[start:stop] = scores["d"], scores["total"]
        start = stop
    best_d = float(all_d.min())
    tied = np.flatnonzero(all_d <= best_d + atol)
    winner = tied[np.argmax(all_total[tied]) if kind == "goods" else np.argmin(all_total[tied])]
    remaining = int(winner)
    owners = [0] * m
    for g in range(m - 1, -1, -1):
        owners[g], remaining = remaining % n, remaining // n
    return ReferenceResult("enumerate", "optimal", owners, best_d, float(all_total[winner]),
                           True, time.perf_counter() - started)


def round_robin(weights):
    weights = validate_weights(weights)
    owners = np.full(weights.shape[1], -1, dtype=int)
    for turn in range(len(owners)):
        i = turn % len(weights)
        remaining = np.flatnonzero(owners < 0)
        owners[remaining[weights[i, remaining].argmax()]] = i
    return owners


def greedy_chores(weights):
    return row_shares(weights).argmin(axis=0)
