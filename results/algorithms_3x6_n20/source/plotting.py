"""Notebook-friendly figures; all computations live outside notebook cells."""
from __future__ import annotations

import numpy as np

from .analysis import paired_differences, summarize
from .data import generate_instances, row_shares

GROUP_LABEL = {"paper": "Статья", "uniform": "1–10"}


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


def plot_results(result, *, metrics=("ef1", "sd_ef1"), n_boot=3000):
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
