"""Memory-bounded enumeration and an optional MILP oracle for larger E1 cases."""
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
    mip_gap: float | None = None
    dual_bound: float | None = None

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


def enumerate_reference(weights, kind, config=None, *, row_total=100.0, atol=1e-9):
    config = config or ReferenceConfig(method="enumerate")
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


def milp_reference(weights, kind, config=None, *, row_total=100.0, atol=1e-9):
    """Optimize SW/C subject to EF1; EF1 is feasible for additive nonnegative inputs.

    x[i,g] assigns items. r[i,j,g] chooses at most one item to hypothetically
    remove for that pair, with r <= x[j,g] (goods) or r <= x[i,g] (chores).
    The returned allocation is always checked by the independent checker.
    """
    validate_kind(kind)
    weights = validate_weights(weights)
    config = config or ReferenceConfig(method="milp")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except ImportError:
        return ReferenceResult("milp", "unavailable", message="Установите пакет с extra solver: pip install -e '.[solver]'")
    started = time.perf_counter()
    n, m = weights.shape
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    nx, size = n * m, n * m + len(pairs) * m
    objective = np.zeros(size)
    objective[:nx] = (row_total * row_shares(weights)).ravel() * (-1 if kind == "goods" else 1)
    rows, columns, values, low, high = [], [], [], [], []

    def constraint(coefficients, lb=-np.inf, ub=np.inf):
        row = len(low)
        for column, value in coefficients.items():
            if value:
                rows.append(row); columns.append(column); values.append(value)
        low.append(lb); high.append(ub)

    for g in range(m):
        constraint({i * m + g: 1 for i in range(n)}, 1, 1)
    for p, (i, j) in enumerate(pairs):
        rstart = nx + p * m
        removal_owner = j if kind == "goods" else i
        constraint({rstart + g: 1 for g in range(m)}, ub=1)
        for g in range(m):
            constraint({rstart + g: 1, removal_owner * m + g: -1}, ub=0)
        sign = 1 if kind == "goods" else -1
        coefficients = {}
        for g in range(m):
            coefficients[j * m + g] = sign * weights[i, g]
            coefficients[i * m + g] = -sign * weights[i, g]
            coefficients[rstart + g] = -weights[i, g]
        constraint(coefficients, ub=0)
    matrix = coo_array((np.array(values, dtype=float),
                        (np.array(rows, dtype=np.int32), np.array(columns, dtype=np.int32))),
                       shape=(len(low), size)).tocsc()
    solution = milp(objective, integrality=np.ones(size), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, low, high),
                    options={"time_limit": config.time_limit, "mip_rel_gap": 0.0})
    result = ReferenceResult("milp", "limit" if solution.status == 1 else "error",
                             seconds=time.perf_counter() - started, message=str(solution.message))
    if solution.x is None:
        return result
    assignments = solution.x[:nx].reshape(n, m)
    if not np.allclose(assignments, np.rint(assignments), atol=1e-6, rtol=0) or not np.allclose(assignments.sum(axis=0), 1):
        result.status, result.message = "invalid", "Решатель вернул нецелочисленное назначение"
        return result
    owners = assignments.argmax(axis=0)
    checked = allocation_scores(weights, owners, kind, row_total=row_total, atol=atol)
    if checked["d"][0] > atol:
        result.status, result.message = "invalid", "Независимая проверка EF1 отклонила решение MILP"
        return result
    result.owners = owners.tolist()
    result.best_d, result.best_total = 0.0, float(checked["total"][0])
    result.certified = bool(solution.success and solution.status == 0)
    result.status = "optimal" if result.certified else "feasible_limit"
    for field in ("mip_gap", "mip_dual_bound"):
        value = getattr(solution, field, None)
        if value is not None and np.isfinite(value):
            setattr(result, "dual_bound" if field == "mip_dual_bound" else field, float(value) * (-1 if field == "mip_dual_bound" and kind == "goods" else 1))
    return result


def exact_reference(weights, kind, config=None, *, row_total=100.0, atol=1e-9):
    config = config or ReferenceConfig()
    validate_kind(kind)
    weights = validate_weights(weights)
    method = config.method
    if method == "auto":
        method = "enumerate" if len(weights) ** weights.shape[1] <= config.max_allocations else "milp"
    if method == "skip":
        return ReferenceResult("skip", "skipped", message="Точный ориентир отключён настройкой")
    solver = enumerate_reference if method == "enumerate" else milp_reference
    return solver(weights, kind, config, row_total=row_total, atol=atol)


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
