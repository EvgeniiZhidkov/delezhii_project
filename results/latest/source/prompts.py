"""E1 prompt and strict JSON validation; no dependence on active notebook size."""
from __future__ import annotations

import json
import numpy as np

from .data import validate_weights
from .metrics import validate_kind

PROMPT_VERSION = "e1-paired-v2"
LEGACY_PROMPT_VERSION = "e1-paired-v1"
LEGACY_PROMPT_VARIANT = "ef1_welfare"
PROMPT_VARIANTS = {
    "free": "Без метрики",
    "ef1": "EF1",
    "prop": "PROP",
    "welfare": "SW/C",
    LEGACY_PROMPT_VARIANT: "EF1 + SW/C",
}


def model_method(label, variant, variants):
    if variants == (LEGACY_PROMPT_VARIANT,):
        return label
    return f"{label} · {PROMPT_VARIANTS[variant]}"


def unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Повтор ключа JSON")
    return result


def reject_json_constant(constant):
    raise ValueError(f"Недопустимое значение JSON: {constant}")


def read_json(text):
    try:
        return json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_json_constant)
    except (TypeError, RecursionError) as exc:
        raise ValueError("Некорректный JSON") from exc


def _goal(kind, variant, row_total, n_agents):
    if variant == "free":
        return "Распредели предметы так, как считаешь справедливым."

    goods = kind == "goods"
    objective = ("максимизируй суммарную полезность полученных благ" if goods else
                 "минимизируй суммарные затраты на назначенные работы")
    if variant == "welfare":
        return f"Цель — {objective}. Суммируй оценки назначенных наборов от лица их получателей."

    if variant == "prop":
        share = f"{row_total:g}/{n_agents}"
        ratio = (f"оценка назначенного ему набора / ({share})" if goods else
                 f"({share}) / стоимость назначенного ему набора")
        zero = " Если стоимость набора равна нулю, его показатель равен 1." if not goods else ""
        return (
            f"Цель — максимизировать приближение к пропорциональности (PROP). Пропорциональная доля равна {share}. "
            f"Для каждого участника посчитай показатель: {ratio}. "
            "Значения выше 1 считай равными 1."
            f"{zero} Максимизируй наименьший показатель среди участников."
        )

    comparison = ("Обозначим через X_i набор участника i. Для каждой пары разных участников i и j "
                  "считай обе оценки от лица i. ")
    if goods:
        removal = "Удали из X_j самое ценное для i благо, если X_j непуст."
        ratio = "оценка X_i / оценка оставшейся части X_j"
        residual = "max(0, оценка X_j после удаления одного блага − оценка X_i)"
    else:
        removal = "Удали из X_i самую затратную для i работу, если X_i непуст."
        ratio = "стоимость X_j / стоимость оставшейся части X_i"
        residual = "max(0, оценка X_i после удаления одной работы − оценка X_j)"

    if variant == "ef1":
        return (
            f"Цель — максимизировать приближение к отсутствию зависти с точностью до одного предмета (EF1). "
            f"{comparison}{removal} Если удалять нечего, используй исходный набор. "
            f"Показатель для пары: {ratio}. Значения выше 1 считай равными 1. "
            "При нулевом знаменателе показатель равен 1. Максимизируй наименьший показатель по всем парам. "
            "Если пар нет, общий показатель равен 1."
        )

    return (
        f"{comparison}{removal} Остаток зависти равен {residual}. "
        "Если удалять нечего, используй исходный набор. Если все остатки равны нулю, "
        "делёж удовлетворяет EF1. Сначала минимизируй максимальный остаток зависти. "
        f"При равном остатке {objective}."
    )


def make_prompt(weights, kind, *, row_total=100.0, variant=LEGACY_PROMPT_VARIANT):
    validate_kind(kind)
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Неизвестный вариант промпта: {variant}")
    values = validate_weights(weights)
    n, m = values.shape
    agents, items = [f"A{i+1}" for i in range(n)], [f"G{g+1}" for g in range(m)]
    table = "\n".join(agent + " | " + " | ".join(f"{v:g}" for v in row)
                      for agent, row in zip(agents, values))
    if kind == "goods":
        task = "Распредели неделимые блага. Чем больше оценка набора, тем он лучше."
    else:
        task = "Распредели неделимые обязанности. Чем больше оценка набора, тем он обременительнее."
    example = json.dumps({"status": "allocation", "bundles": {agent: ["ID", "..."] for agent in agents}}, ensure_ascii=False)
    return (
        f"{task}\nНазначь каждую позицию ровно одному участнику. Пустые наборы разрешены.\n"
        "В таблице исходные веса. Оценка набора S для участника i равна "
        f"{row_total:g} × (сумма весов позиций S в строке i) / (сумма всех весов в строке i).\n\n"
        f"Участник | {' | '.join(items)}\n{table}\n\n"
        f"{_goal(kind, variant, row_total, n)}\n\n"
        f"Верни один JSON без пояснений и без Markdown по схеме {example}. "
        f"В списках используй только реальные ID от {items[0]} до {items[-1]}. Для пустого набора используй []."
    )


def parse_allocation(raw, n_agents, n_items):
    if not isinstance(raw, str):
        raise ValueError("Пустой или нетекстовый ответ")
    answer = read_json(raw)
    if not isinstance(answer, dict) or set(answer) != {"status", "bundles"} or answer["status"] != "allocation":
        raise ValueError("Ожидаются ровно status=allocation и bundles")
    agents = [f"A{i+1}" for i in range(n_agents)]
    bundles = answer["bundles"]
    if not isinstance(bundles, dict) or set(bundles) != set(agents):
        raise ValueError("Неверный список участников")
    index = {f"G{g+1}": g for g in range(n_items)}
    owners = np.full(n_items, -1, dtype=int)
    for i, agent in enumerate(agents):
        if not isinstance(bundles[agent], list):
            raise ValueError("Набор участника должен быть списком")
        for item in bundles[agent]:
            if not isinstance(item, str) or item not in index:
                raise ValueError("Неизвестный ID позиции")
            g = index[item]
            if owners[g] != -1:
                raise ValueError("Позиция указана несколько раз")
            owners[g] = i
    if (owners == -1).any():
        raise ValueError("Некоторые позиции пропущены")
    return owners
