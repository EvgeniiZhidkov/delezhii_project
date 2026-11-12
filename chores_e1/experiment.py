"""Run E1 and replay saved raw answers through the current independent checker."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from .backends import CodexBackend
from .config import ExperimentConfig
from .data import generate_instances
from .metrics import allocation_scores, answer_metrics
from .prompts import PROMPT_VERSION, make_prompt, parse_allocation, read_json
from .reference import ReferenceResult, exact_reference, greedy_chores, round_robin


@dataclass
class ExperimentResult:
    directory: Path
    config: ExperimentConfig
    rows: pd.DataFrame
    references: pd.DataFrame
    complete: bool = True


def emit(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def trial_plan(config, instances):
    for instance in instances:
        for kind in ("goods", "chores"):
            yield instance, kind, "reference", 0, None
            yield instance, kind, "heuristic", 0, None
            for model in config.models:
                for repeat in range(1, config.repeats + 1):
                    yield instance, kind, "model", repeat, model


def key_for(instance, kind, source, repeat, model):
    return [*instance.key, kind, source, repeat, model.label if model else None]


def score_record(record, instance, config, reference):
    source, kind = record["source"], record["kind"]
    owners, format_error = record.get("owners"), None
    if source == "model":
        if record["backend"].get("error") is not None:
            return None
        try:
            owners = parse_allocation(record["backend"].get("raw"), instance.setting.n_agents, instance.setting.n_items)
        except ValueError as exc:
            owners, format_error = None, str(exc)
    if source == "reference" and owners is None:
        return None  # skipped oracle is not a failed allocation
    method = record["method"]
    return {"setting": instance.setting.label, "n_agents": instance.setting.n_agents,
            "n_items": instance.setting.n_items, "base_id": instance.base_id, "group": instance.group,
            "kind": kind, "source": source, "method": method, "repeat": record["repeat"],
            "reference_status": reference.status, "reference_certified": reference.certified,
            "format_error": format_error, "seconds": record.get("backend", {}).get("seconds"),
            "usage": record.get("backend", {}).get("usage"),
            **answer_metrics(instance.weights, owners, kind, reference,
                             row_total=config.row_total, atol=config.atol)}


def validate_reference(reference, instance, kind, config):
    """Validate a saved reference before using its totals during replay."""
    if reference.certified and (reference.status != "optimal" or reference.owners is None):
        raise ValueError("Несогласованный статус точного ориентира")
    if reference.owners is None:
        return
    checked = allocation_scores(instance.weights, reference.owners, kind,
                                row_total=config.row_total, atol=config.atol)
    if len(checked["d"]) != 1 or reference.best_d is None or reference.best_total is None:
        raise ValueError("Некорректная запись точного ориентира")
    if not np.isfinite([reference.best_d, reference.best_total]).all():
        raise ValueError("Неконечные показатели ориентира")
    if checked["d"][0] > reference.best_d + config.atol or not np.isclose(
        checked["total"][0], reference.best_total, rtol=0, atol=config.atol
    ):
        raise ValueError("Оценки ориентира не совпадают с сохранённым распределением")


def run_experiment(config=None, *, backend=None, progress=None):
    config = config or ExperimentConfig()
    instances = generate_instances(config)
    # Fail before solving baselines if a requested backend is unavailable.
    if config.models and backend is None:
        backend = CodexBackend()
    directory = Path(config.output_dir) / ("e1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    source_dir = directory / "source"
    source_dir.mkdir()
    hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(path, source_dir / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ("numpy", "pandas", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = {"schema": 2, "config": config.record(), "prompt_version": PROMPT_VERSION,
                "python": platform.python_version(), "packages": versions, "source_sha256": hashes,
                "backend_version": getattr(backend, "version", None)}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, references = [], []
    reference = None
    plan_size = len(instances) * 2 * (2 + len(config.models) * config.repeats)
    with (directory / "journal.jsonl").open("x", encoding="utf-8") as journal:
        emit(journal, {"type": "config", **manifest})
        for position, (instance, kind, source, repeat, model) in enumerate(trial_plan(config, instances), 1):
            record = {"type": "trial", "key": key_for(instance, kind, source, repeat, model),
                      **instance.record(), "kind": kind, "source": source, "repeat": repeat}
            if source == "reference":
                reference = exact_reference(instance.weights, kind, config.reference,
                                            row_total=config.row_total, atol=config.atol)
                validate_reference(reference, instance, kind, config)
                record.update(reference=reference.record(), owners=reference.owners,
                              method="Точный ориентир")
                references.append({**instance.record(), "kind": kind, **reference.record()})
            elif source == "heuristic":
                owners = round_robin(instance.weights) if kind == "goods" else greedy_chores(instance.weights)
                record.update(owners=owners.tolist(), method="Эвристика")
            else:
                prompt = make_prompt(instance.weights, kind, row_total=config.row_total)
                call_dir = directory / "calls" / f"{position:06d}"
                reply = backend(prompt, model, call_dir)
                record.update(method=model.label, prompt=prompt, model_config=model.__dict__, backend=reply.record())
            # Preserve the original answer before any parsing, including backend failures.
            emit(journal, record)
            if source == "model" and record["backend"].get("error") is not None:
                raise RuntimeError(f"{record['backend']['error']}\nСохранён незавершённый прогон: {directory}")
            scored = score_record(record, instance, config, reference)
            if scored is not None:
                rows.append(scored)
            if progress:
                progress(position, plan_size, record["key"])
        emit(journal, {"type": "complete", "trials": plan_size})
    result = ExperimentResult(directory, config, pd.DataFrame(rows), pd.DataFrame(references))
    result.rows.to_csv(directory / "results.csv", index=False)
    result.references.drop(columns=["weights"]).to_csv(directory / "references.csv", index=False)
    return result


def analyze_file(path, *, allow_partial=False):
    """Reject mismatched inputs, prompts, ordering and incomplete logs by default."""
    path = Path(path)
    if path.is_dir():
        path = path / "journal.jsonl"
    with path.open(encoding="utf-8") as stream:
        config_record = read_json(stream.readline())
        if not isinstance(config_record, dict) or config_record.get("type") != "config" or config_record.get("schema") != 2:
            raise ValueError("Ожидается журнал E1 schema=2; старый пилот сохранён в archive/")
        if config_record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Версия промпта отличается")
        config = ExperimentConfig.from_dict(config_record["config"])
        instances = generate_instances(config)
        rows, references = [], []
        reference = None
        count, complete = 0, False
        for instance, kind, source, repeat, model in trial_plan(config, instances):
            line = stream.readline()
            if not line:
                break
            record = read_json(line)
            if record.get("type") != "trial" or record.get("key") != key_for(instance, kind, source, repeat, model):
                raise ValueError("Ответ не соответствует порядку запланированных запросов")
            if any(record.get(field) != value for field, value in instance.record().items()):
                raise ValueError("Записанный вход не совпадает с генератором")
            if any(record.get(field) != value for field, value in {"kind": kind, "source": source, "repeat": repeat}.items()):
                raise ValueError("Тип ответа отличается от плана")
            if source == "reference":
                reference = ReferenceResult(**{
                    key: value for key, value in record["reference"].items()
                    if key in ReferenceResult.__dataclass_fields__
                })
                validate_reference(reference, instance, kind, config)
                if record.get("owners") != reference.owners:
                    raise ValueError("Назначение ориентира не совпадает с записью решателя")
                if record.get("method") != "Точный ориентир":
                    raise ValueError("Неверная метка ориентира")
                references.append({**instance.record(), "kind": kind, **reference.record()})
            elif source == "heuristic":
                expected = round_robin(instance.weights) if kind == "goods" else greedy_chores(instance.weights)
                if record.get("method") != "Эвристика" or record.get("owners") != expected.tolist():
                    raise ValueError("Эвристика отличается от сохранённого протокола")
            elif source == "model":
                if record.get("prompt") != make_prompt(instance.weights, kind, row_total=config.row_total) or record.get("model_config") != model.__dict__:
                    raise ValueError("Промпт или модель отличаются от плана")
                if record.get("method") != model.label:
                    raise ValueError("Метка модели отличается от плана")
                if record.get("backend", {}).get("error") is not None:
                    if allow_partial:
                        break
                    raise ValueError("Прогон прерван ошибкой CLI")
            scored = score_record(record, instance, config, reference)
            if scored is not None:
                rows.append(scored)
            count += 1
        else:
            ending = stream.readline()
            if ending:
                marker = read_json(ending)
                complete = marker == {"type": "complete", "trials": count}
                if not complete or stream.read().strip():
                    raise ValueError("Неверное завершение журнала или лишние ответы")
        if not complete and not allow_partial:
            raise ValueError("Прогон не завершён; для диагностики задайте allow_partial=True")
    return ExperimentResult(path.parent, config, pd.DataFrame(rows), pd.DataFrame(references), complete)
