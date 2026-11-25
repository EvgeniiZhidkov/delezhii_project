"""Notebook-friendly figures; all computations live outside notebook cells."""
from __future__ import annotations

import numpy as np

from .analysis import paired_differences, summarize
from .data import generate_instances, row_shares

GROUP_LABEL = {"paper": "Статья", "uniform": "1–10", "no_ef": "Без EF", "no_efx": "Без EFX для обязанностей"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_inputs(config):
    plt = _plt()
    instances = generate_instances(config)
    sample = instances[0]
    n, m = sample.weights.shape
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    heat = axes[0].imshow(config.row_total * row_shares(sample.weights), cmap="YlGnBu", aspect="auto")
    axes[0].set(title=f"Вход {sample.setting.label}, {sample.group}", xlabel="Предмет", ylabel="Участник",
                xticks=range(m), xticklabels=[f"G{g+1}" for g in range(m)],
                yticks=range(n), yticklabels=[f"A{i+1}" for i in range(n)])
    fig.colorbar(heat, ax=axes[0], label=f"Баллы из {config.row_total:g}")
    totals = np.concatenate([x.weights.sum(axis=1) for x in instances
                             if x.setting == sample.setting and x.group == sample.group])
    axes[1].hist(totals, bins=min(12, max(1, len(totals))), color="#11786f", edgecolor="white")
    axes[1].set(title="Суммы строк до нормировки", xlabel="Сумма весов", ylabel="Число участников")
    return fig


def _draw(ax, frame, groups, label, offset, color):
    positions = {group: i for i, group in enumerate(groups)}
    for subset, style in (([g for g in groups if not g.startswith("r=")], "o"),
                          ([g for g in groups if g.startswith("r=")], "o-")):
        points = frame.set_index("group").reindex(subset).dropna(subset=["mean"])
        if points.empty:
            continue
        x = np.array([positions[g] for g in points.index]) + offset
        # Missing intervals (one defined instance) are drawn as points only.
        lo, hi = points["low"].to_numpy(), points["high"].to_numpy()
        mean = points["mean"].to_numpy()
        errors = np.maximum([mean - lo, hi - mean], 0)
        errors[:, ~np.isfinite(errors).all(axis=0)] = 0
        ax.errorbar(x, mean, yerr=errors, fmt=style, color=color, capsize=3, label=label)
        label = "_nolegend_"


def plot_rates(result, *, metrics=("ef1", "sd_ef1"), n_boot=3000):
    plt = _plt()
    from matplotlib.ticker import PercentFormatter
    stats = summarize(result.rows, n_base=result.config.n_base, seed=result.config.seed + 1, n_boot=n_boot)
    figures = []
    names = {"ef1": "Доля допустимых EF1-ответов", "sd_ef1": "Доля допустимых SD-EF1-ответов",
             "sd_ef": "Доля допустимых SD-EF-ответов", "sd_d": "Нарушение SD-EF1", "d": "Нарушение EF1"}
    for setting in result.config.settings:
        for metric in metrics:
            selected = stats[(stats["setting"] == setting.label) & (stats["metric"] == metric)]
            if selected.empty:
                continue
            methods = list(selected["method"].unique())
            is_rate = metric in {"valid", "ef1", "sd_ef", "sd_ef1", "sd_ef_pairs", "sd_ef1_pairs"}
            colors = plt.get_cmap("tab10")
            offsets = np.linspace(-0.14, 0.14, len(methods))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
            for ax, kind in zip(axes, ("goods", "chores")):
                for i, (method, offset) in enumerate(zip(methods, offsets)):
                    frame = selected[(selected["method"] == method) & (selected["kind"] == kind)]
                    _draw(ax, frame, result.config.groups, method, offset, colors(i))
                ax.set(title=f"{setting.label} · {'Блага' if kind == 'goods' else 'Обязанности'}",
                       xticks=range(len(result.config.groups)),
                       xticklabels=[GROUP_LABEL.get(g, g) for g in result.config.groups])
                ax.grid(axis="y", alpha=0.25)
                if is_rate:
                    ax.set_ylim(-0.06, 1.08)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
            axes[0].set_ylabel(names.get(metric, metric))
            axes[1].legend(frameon=False, fontsize=8)
            figures.append(fig)
    return figures


# Six spokes and their ordering match Figure 1 of Cookson–Ebadian–Shah.
PAPER_METRICS = ("prop_alpha", "ef_alpha", "ef1_alpha", "po_alpha", "sw_ratio", "mms_alpha")
EXTENDED_METRICS = ("prop_alpha", "ef_alpha", "ef1_alpha", "efx_alpha", "sd_ef1",
                    "po_alpha", "sw_ratio", "nw_ratio", "mms_alpha")
METRIC_LABELS = {"prop_alpha": "PROP", "ef_alpha": "EF", "ef1_alpha": "EF1",
                 "mms_alpha": "MMS", "po_alpha": "PO", "sw_ratio": "SW",
                 "nw_ratio": "MNW", "efx_alpha": "EFX", "sd_ef1": "SD-EF1",
                 "prop1_alpha": "PROP1"}
EFFICIENCY = {"po_alpha", "sw_ratio", "nw_ratio"}
SCALING_METRICS = {
    "ef1_alpha": "Приближение\nк EF1",
    "sw_ratio": "Эффективность\n(SW / C)",
    "sd_ef1": "Доля SD-EF1",
    "efx_alpha": "Приближение\nк EFX+",
    "efx": "Доля EFX+",
    "nw_ratio_lower": "Доля MNW\n(нижняя оценка)",
    "ef_alpha": "Приближение\nк EF",
    "ef1": "Доля EF1",
}
PAPER_SCALING_METRICS = ("ef1_alpha", "sw_ratio", "sd_ef1", "efx_alpha", "efx", "nw_ratio_lower")


def radar_table(result, *, group=None, metrics=EXTENDED_METRICS, n_boot=None):
    """Means after repeats, then equal group weights, then base matrices.

    n_defined/expected expose missing exact benchmarks instead of replacing
    them by zero. Invalid model answers have explicit zero scores.
    """
    from .analysis import _interval, per_instance
    import pandas as pd
    base = per_instance(result.rows).reset_index()
    if group is not None:
        if group not in result.config.groups:
            raise ValueError(f"Неизвестная группа: {group}")
        base = base[base["group"] == group]
    if n_boot is not None:
        draws = np.random.default_rng(result.config.seed + 1).integers(
            result.config.n_base, size=(n_boot, result.config.n_base))
    records = []
    for key, frame in base.groupby(["setting", "method", "kind"], sort=False):
        means = frame.groupby("base_id")[list(metrics)].mean()
        record = dict(zip(["setting", "method", "kind"], key))
        for metric in metrics:
            defined = int(frame[metric].notna().sum())
            row = {**record, "metric": metric, "mean": means[metric].mean(),
                   "n_defined": defined, "n_expected": len(frame)}
            if n_boot is not None:
                _, low, high, _ = _interval(means[metric].reindex(range(result.config.n_base)), draws)
                row.update(low=low, high=high)
            records.append(row)
    return pd.DataFrame(records)


def scaling_table(result, *, group=None):
    """Per-size means, coverage and 95% bootstrap intervals over base matrices."""
    metrics = (*SCALING_METRICS, "nw_ratio")
    stats = radar_table(result, group=group, metrics=metrics, n_boot=3000)
    if stats.empty:
        return stats
    sizes = result.rows[["setting", "n_agents", "n_items"]].drop_duplicates()
    return sizes.merge(stats, on="setting", validate="one_to_many")


def _scaling_figure(panels, metrics, title, scope, colors):
    from matplotlib.ticker import PercentFormatter

    plt = _plt()
    counts = {count for *_, frame in panels for count in frame.groupby("setting")["n_expected"].max()}
    count_label = f"Матриц на точку — {next(iter(counts))}" if len(counts) == 1 else "Число матриц в скобках"
    fig, axes = plt.subplots(len(metrics), len(panels), squeeze=False, sharex="col", sharey="row",
                             layout="constrained", figsize=(max(6, 3.5 * len(panels)), 2 * len(metrics) + 0.8))
    handles = {}
    for column, (x, title_suffix, x_label, frame) in enumerate(panels):
        matrix_counts = frame.groupby(x)["n_expected"].max()
        positions = matrix_counts.index
        ticks = [str(value) if len(counts) == 1 else f"{value}\n({count})"
                 for value, count in matrix_counts.items()]
        axes[0, column].set_title(title_suffix)
        axes[-1, column].set_xticks(positions, ticks)
        axes[-1, column].set_xlabel(x_label)
        for row, metric in enumerate(metrics):
            ax = axes[row, column]
            series = frame[frame["metric"] == metric]
            for method, observations in series.groupby("method", sort=False):
                values = observations.set_index(x).reindex(positions)
                mean = values["mean"].to_numpy(float)
                if not np.isfinite(mean).any():
                    continue
                errors = np.maximum([mean - values["low"], values["high"] - mean], 0)
                handles[method] = ax.errorbar(positions, mean, yerr=errors, fmt="o-", markersize=4,
                                              capsize=3, label=method, color=colors[method])
            if not series["mean"].notna().any():
                ax.text(0.5, 0.5, "н/д", transform=ax.transAxes, ha="center")
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=0.25)
            if column == 0:
                ax.set_ylabel(SCALING_METRICS[metric])
    fig.suptitle(f"{title}\n{scope} · {count_label} · 95% бутстрэп-интервалы")
    if handles:
        fig.legend(handles=list(handles.values()), loc="outside lower center",
                   ncol=min(4, len(handles)), frameon=False)
    return fig


def plot_scaling(result, *, group=None, paper=False, stats=None):
    """Compare n and m. The paper layout shows the two slices through 5×15."""
    stats = scaling_table(result, group=group) if stats is None else stats
    if stats.empty:
        return {}
    if paper and result.config.models:
        rows = result.rows
        other_prompts = rows.loc[(rows["source"] == "model") & (rows["prompt_variant"] != "free"), "method"]
        stats = stats[~stats["method"].isin(other_prompts)]
    methods = stats["method"].unique()
    colors = dict(zip(methods, _plt().get_cmap("tab10")(np.arange(len(methods)) % 10)))
    scope = GROUP_LABEL.get(group, group) if group else "Все группы"
    figures = {}
    for kind, kind_label in (("goods", "Блага"), ("chores", "Обязанности")):
        selected = stats[stats["kind"] == kind]
        metrics = [metric for metric in (PAPER_SCALING_METRICS if paper else SCALING_METRICS)
                   if kind == "goods" or metric != "nw_ratio_lower"]
        paper_panels = []
        for x, fixed, symbol, fixed_symbol, fixed_at, x_label in (
            ("n_agents", "n_items", "n", "m", 15, "Число участников n"),
            ("n_items", "n_agents", "m", "n", 5, "Число позиций m"),
        ):
            panels = [(x, f"{fixed_symbol} = {value}", x_label, frame)
                      for value, frame in selected.groupby(fixed)
                      if frame[x].nunique() > 1 and (not paper or value == fixed_at)]
            if paper:
                paper_panels.extend(panels)
            elif panels:
                figures[f"scaling_{symbol}_{kind}"] = _scaling_figure(
                    panels, metrics, f"{kind_label} · зависимость от {symbol}", scope, colors)
        if len(paper_panels) == 2:
            figures[f"scaling_paper_{kind}"] = _scaling_figure(
                paper_panels, metrics, f"{kind_label} · сравнение по n и m", scope, colors)
    return figures


def plot_results(result, *, metrics=PAPER_METRICS, group=None, include_reference=False):
    """Paper-style radar panels: one method per panel, separate domains/sizes."""
    plt = _plt()
    from matplotlib.patches import Patch
    from .prompts import LEGACY_PROMPT_VARIANT, PROMPT_VARIANTS, model_method
    stats = radar_table(result, group=group, metrics=metrics)
    figures = []
    if stats.empty:
        return figures
    count = len(metrics)
    angles = (np.arange(count) * 2 * np.pi / count - np.pi / count) % (2 * np.pi)
    variants = result.config.prompt_variants
    model_titles = {}
    for model in result.config.models:
        for variant in variants:
            title = model.model or model.label
            if variants != (LEGACY_PROMPT_VARIANT,):
                title += f"\n{PROMPT_VARIANTS[variant]}"
            model_titles[model_method(model.label, variant, variants)] = title
    for setting in result.config.settings:
        for kind, kind_label in (("goods", "Блага"), ("chores", "Обязанности")):
            selected = stats[(stats["setting"] == setting.label) & (stats["kind"] == kind)]
            if not include_reference:
                from .reference import REFERENCE_LABEL
                selected = selected[selected["method"] != REFERENCE_LABEL]
            methods = list(selected["method"].unique())
            if not methods:
                continue
            columns = min(4, len(methods))
            rows = int(np.ceil(len(methods) / columns))
            # Reserve room above the axes for both the figure and panel titles.
            header_height = 1.9
            footer_height = 0.65
            height = 4.8 * rows + header_height + footer_height
            fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, height),
                                     subplot_kw={"projection": "polar"}, squeeze=False)
            fig.subplots_adjust(left=0.06, right=0.94, top=1 - header_height / height,
                                bottom=footer_height / height, wspace=0.55, hspace=0.8)
            for ax, method in zip(axes.flat, methods):
                frame = selected[selected["method"] == method].set_index("metric").reindex(metrics)
                values = frame["mean"].to_numpy(float)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                colors = ["#f7bd91" if metric in EFFICIENCY else "#a9d39e" for metric in metrics]
                ax.bar(angles, np.ones(count), width=2*np.pi/count*0.88, bottom=0,
                       color=colors, edgecolor="none", zorder=0)
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0.2, 1.01, 0.2))
                ax.set_yticklabels([f"{value}%" for value in (20, 40, 60, 80, 100)],
                                   fontsize=8, color="#6d746e")
                ax.set_rlabel_position(65)
                labels = []
                for metric, value, (_, row) in zip(metrics, values, frame.iterrows()):
                    percent = f"{value:.0%}" if np.isfinite(value) else "н/д"
                    partial = "*" if 0 < row["n_defined"] < row["n_expected"] else ""
                    labels.append(f"{METRIC_LABELS.get(metric, metric)}\n({percent}{partial})")
                ax.set_xticks(angles, labels, fontsize=11, fontfamily="DejaVu Serif")
                ax.set_xlim(0, 2 * np.pi)
                ax.tick_params(axis="x", pad=14)
                ax.grid(color="#566254", linestyle="--", linewidth=0.65, alpha=0.35)
                # NaN remains a visible gap: an unavailable metric is never a zero.
                ax.plot(np.r_[angles, angles[0]], np.r_[values, values[0]], color="#161c17", lw=2.2, zorder=3)
                ax.scatter(angles[np.isfinite(values)], values[np.isfinite(values)],
                           color="#161c17", s=12, zorder=4)
                ax.set_title(model_titles.get(method, method), fontsize=13, fontweight="bold", pad=42)
            for ax in list(axes.flat)[len(methods):]:
                ax.set_visible(False)
            scope = GROUP_LABEL.get(group, group) if group else f"все группы ({len(result.config.groups)})"
            instances = result.config.n_base * (1 if group else len(result.config.groups))
            fig.suptitle(f"{kind_label} · {setting.label} · {scope} · {instances} матриц\n"
                         "Средние оценки · 100% — лучшее значение", fontsize=15, y=1 - 0.10 / height)
            fig.legend(handles=[Patch(color="#a9d39e", label="Справедливость"),
                                Patch(color="#f7bd91", label="Эффективность")],
                       loc="lower left", bbox_to_anchor=(0.04, 0.005), ncol=2, frameon=False, fontsize=10)
            note = "SD-EF1 — доля успехов; остальные оси — приближения." if "sd_ef1" in metrics else "Оси — приближения к критериям, а не доли успехов."
            if kind == "chores":
                note += " SW = min C / C; MNW не применяется."
            if frame["mean"].isna().any() or (selected["n_defined"] < selected["n_expected"]).any():
                note += " н/д — нет оценки; * — неполное покрытие."
            fig.text(0.96, 0.025, note, ha="right", va="bottom", fontsize=8, color="#4f5951")
            figures.append(fig)
    return figures


def plot_paired(result, *, metric="ef1", n_boot=3000):
    plt = _plt()
    from matplotlib.ticker import PercentFormatter
    stats = paired_differences(result.rows, metric=metric, n_base=result.config.n_base,
                               seed=result.config.seed + 1, n_boot=n_boot)
    figures = []
    for setting in result.config.settings:
        selected = stats[stats["setting"] == setting.label]
        fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
        methods = list(selected["method"].unique())
        for i, (method, offset) in enumerate(zip(methods, np.linspace(-0.14, 0.14, len(methods)))):
            _draw(ax, selected[selected["method"] == method], result.config.groups, method, offset, plt.get_cmap("tab10")(i))
        ax.axhline(0, color="#506577", linewidth=1)
        ax.set(title=f"{setting.label} · обязанности минус блага · {metric}",
               xticks=range(len(result.config.groups)),
               xticklabels=[GROUP_LABEL.get(g, g) for g in result.config.groups], ylabel="Разница долей, п. п.")
        ax.yaxis.set_major_formatter(PercentFormatter(1, symbol=""))
        ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
        figures.append(fig)
    return figures
