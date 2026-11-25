"""Run E1 and replay saved raw answers through the current independent checker."""
from __future__ import annotations

from dataclasses import dataclass, replace
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
from .evaluation import build_evaluation
from .metrics import allocation_scores, answer_metrics
from .prompts import (LEGACY_PROMPT_VARIANT, LEGACY_PROMPT_VERSION, PROMPT_VERSION,
                      make_prompt, model_method, parse_allocation, read_json)
from .reference import (ReferenceResult, exact_reference, greedy_chores, round_robin,
                        maximum_social_welfare, minimum_social_welfare,
                        REFERENCE_LABEL, RR_LABEL, MIN_SW_LABEL, MAX_SW_LABEL, MNW_LABEL)


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


JOURNAL_SCHEMA = 4


def trial_plan(config, instances, *, schema=JOURNAL_SCHEMA):
    for instance in instances:
        for kind in ("goods", "chores"):
            yield instance, kind, "reference", 0, None, None
            if schema == 2:
                yield instance, kind, "heuristic", 0, None, None
            else:
                yield instance, kind, "round_robin", 0, None, None
                yield instance, kind, "social_welfare", 0, None, None
                if kind == "goods":
                    yield instance, kind, "mnw", 0, None, None
            for model in config.models:
                for repeat in range(1, config.repeats + 1):
                    for variant in config.prompt_variants:
                        yield instance, kind, "model", repeat, model, variant


def key_for(instance, kind, source, repeat, model, variant, *, schema=JOURNAL_SCHEMA):
    key = [*instance.key, kind, source, repeat, model.label if model else None]
    if schema >= 4 and source == "model":
        key.append(variant)
    return key


def score_record(record, instance, config, reference, evaluation=None):
    source, kind = record["source"], record["kind"]
    owners, format_error = record.get("owners"), None
    if source == "model":
        if record["backend"].get("error") is not None:
            return None
        try:
            owners = parse_allocation(record["backend"].get("raw"), instance.setting.n_agents, instance.setting.n_items)
        except ValueError as exc:
            owners, format_error = None, str(exc)
    if source in {"reference", "mnw"} and owners is None:
        return None  # skipped oracle is not a failed allocation
    method = record["method"]
    if source == "reference":
        method = REFERENCE_LABEL
    elif source == "heuristic":
        method = RR_LABEL if kind == "goods" else MIN_SW_LABEL
    variant = record.get("prompt_variant", LEGACY_PROMPT_VARIANT) if source == "model" else None
    return {"setting": instance.setting.label, "n_agents": instance.setting.n_agents,
            "n_items": instance.setting.n_items, "base_id": instance.base_id, "group": instance.group,
            "kind": kind, "source": source, "method": method, "repeat": record["repeat"],
            "prompt_variant": variant,
            "reference_status": reference.status, "reference_certified": reference.certified,
            "format_error": format_error, "seconds": record.get("backend", {}).get("seconds"),
            "usage": record.get("backend", {}).get("usage"),
            **answer_metrics(instance.weights, owners, kind, reference,
                             row_total=config.row_total, atol=config.atol, evaluation=evaluation)}


def algorithm_answer(weights, kind, source, evaluation):
    if source == "round_robin":
        return round_robin(weights, kind), RR_LABEL
    if source == "social_welfare":
        if kind == "goods":
            return maximum_social_welfare(weights), MAX_SW_LABEL
        return minimum_social_welfare(weights), MIN_SW_LABEL
    if source == "mnw":
        if evaluation.owners is None or evaluation.best_nw <= 0:
            return None, MNW_LABEL
        winner = int(np.argmax(evaluation.nash_welfare))
        return evaluation.owners[winner], MNW_LABEL
    raise ValueError(f"Неизвестный алгоритм: {source}")


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
    manifest = {"schema": JOURNAL_SCHEMA, "config": config.record(), "prompt_version": PROMPT_VERSION,
                "python": platform.python_version(), "packages": versions, "source_sha256": hashes,
                "backend_version": getattr(backend, "version", None)}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, references = [], []
    reference = None
    model_calls = len(config.models) * len(config.prompt_variants) * config.repeats
    plan_size = len(instances) * (7 + 2 * model_calls)
    evaluation = None
    with (directory / "journal.jsonl").open("x", encoding="utf-8") as journal:
        emit(journal, {"type": "config", **manifest})
        for position, (instance, kind, source, repeat, model, variant) in enumerate(trial_plan(config, instances), 1):
            record = {"type": "trial", "key": key_for(instance, kind, source, repeat, model, variant),
                      **instance.record(), "kind": kind, "source": source, "repeat": repeat}
            if source == "reference":
                evaluation = build_evaluation(instance.weights, kind, config.reference)
                reference = exact_reference(instance.weights, kind, config.reference,
                                            row_total=config.row_total, atol=config.atol)
                validate_reference(reference, instance, kind, config)
                record.update(reference=reference.record(), owners=reference.owners,
                              method=REFERENCE_LABEL)
                references.append({**instance.record(), "kind": kind, **reference.record()})
            elif source in {"round_robin", "social_welfare", "mnw"}:
                owners, method = algorithm_answer(instance.weights, kind, source, evaluation)
                record.update(owners=owners.tolist() if owners is not None else None, method=method)
            else:
                prompt = make_prompt(instance.weights, kind, row_total=config.row_total, variant=variant)
                call_dir = directory / "calls" / f"{position:06d}"
                reply = backend(prompt, model, call_dir)
                record.update(method=model_method(model.label, variant, config.prompt_variants),
                              prompt=prompt, prompt_variant=variant, model_config=model.__dict__, backend=reply.record())
            # Preserve the original answer before any parsing, including backend failures.
            emit(journal, record)
            if source == "model" and record["backend"].get("error") is not None:
                raise RuntimeError(f"{record['backend']['error']}\nСохранён незавершённый прогон: {directory}")
            scored = score_record(record, instance, config, reference, evaluation)
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
        if not isinstance(config_record, dict) or config_record.get("type") != "config" or config_record.get("schema") not in {2, 3, JOURNAL_SCHEMA}:
            raise ValueError("Ожидается журнал E1 схемы 2, 3 или 4")
        schema = config_record["schema"]
        prompt_version = PROMPT_VERSION if schema == JOURNAL_SCHEMA else LEGACY_PROMPT_VERSION
        if config_record.get("prompt_version") != prompt_version:
            raise ValueError("Версия промпта отличается")
        config = ExperimentConfig.from_dict(config_record["config"])
        if schema < JOURNAL_SCHEMA and config.prompt_variants != (LEGACY_PROMPT_VARIANT,):
            raise ValueError("В старом журнале допустим только промпт e1-paired-v1")
        instances = generate_instances(config)
        rows, references = [], []
        reference = None
        evaluation = None
        count, complete = 0, False
        for instance, kind, source, repeat, model, variant in trial_plan(config, instances, schema=schema):
            line = stream.readline()
            if not line:
                break
            record = read_json(line)
            expected_key = key_for(instance, kind, source, repeat, model, variant, schema=schema)
            if record.get("type") != "trial" or record.get("key") != expected_key:
                raise ValueError("Ответ не соответствует порядку запланированных запросов")
            if any(record.get(field) != value for field, value in instance.record().items()):
                raise ValueError("Записанный вход не совпадает с генератором")
            if any(record.get(field) != value for field, value in {"kind": kind, "source": source, "repeat": repeat}.items()):
                raise ValueError("Тип ответа отличается от плана")
            if source == "reference":
                evaluation = build_evaluation(instance.weights, kind, config.reference)
                reference = ReferenceResult(**{
                    key: value for key, value in record["reference"].items()
                    if key in ReferenceResult.__dataclass_fields__
                })
                validate_reference(reference, instance, kind, config)
                if record.get("owners") != reference.owners:
                    raise ValueError("Назначение ориентира не совпадает с записью решателя")
                if record.get("method") != ("Точный ориентир" if schema == 2 else REFERENCE_LABEL):
                    raise ValueError("Неверная метка ориентира")
                references.append({**instance.record(), "kind": kind, **reference.record()})
            elif source == "heuristic":
                expected = round_robin(instance.weights) if kind == "goods" else greedy_chores(instance.weights)
                if record.get("method") != "Эвристика" or record.get("owners") != expected.tolist():
                    raise ValueError("Эвристика отличается от сохранённого протокола")
            elif source in {"round_robin", "social_welfare", "mnw"}:
                expected, method = algorithm_answer(instance.weights, kind, source, evaluation)
                expected = expected.tolist() if expected is not None else None
                if record.get("method") != method or record.get("owners") != expected:
                    raise ValueError("Алгоритм отличается от сохранённого протокола")
            elif source == "model":
                expected_prompt = make_prompt(instance.weights, kind, row_total=config.row_total, variant=variant)
                default_variant = LEGACY_PROMPT_VARIANT if schema < JOURNAL_SCHEMA else None
                if record.get("prompt_variant", default_variant) != variant:
                    raise ValueError("Вариант промпта отличается от плана")
                if record.get("prompt") != expected_prompt or record.get("model_config") != model.__dict__:
                    raise ValueError("Промпт или модель отличаются от плана")
                if record.get("method") != model_method(model.label, variant, config.prompt_variants):
                    raise ValueError("Метка модели отличается от плана")
                if record.get("backend", {}).get("error") is not None:
                    if allow_partial:
                        break
                    raise ValueError("Прогон прерван ошибкой CLI")
            scored = score_record(record, instance, config, reference, evaluation)
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


def compare_saved_run(path, *, output_dir="runs"):
    """Recheck saved LLM replies and add today's baselines on identical inputs.

    The original journal is immutable. The new journal contains algorithm calls
    only; comparison.csv and its provenance manifest describe the joined report.
    """
    saved = analyze_file(path)
    baseline = run_experiment(replace(saved.config, models=(), output_dir=str(output_dir)))
    models = saved.rows[saved.rows["source"] == "model"]
    rows = pd.concat([baseline.rows, models], ignore_index=True)
    provenance = {"kind": "offline_comparison", "source_journal": str((saved.directory / "journal.jsonl").resolve()),
                  "source_sha256": hashlib.sha256((saved.directory / "journal.jsonl").read_bytes()).hexdigest(),
                  "algorithm_journal": "journal.jsonl", "model_calls": 0,
                  "model_rows_reused": len(models), "config": saved.config.record()}
    rows.to_csv(baseline.directory / "comparison.csv", index=False)
    (baseline.directory / "comparison_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return ExperimentResult(baseline.directory, saved.config, rows, baseline.references)
