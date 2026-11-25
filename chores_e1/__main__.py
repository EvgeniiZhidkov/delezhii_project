"""python -m chores_e1 run --config configs/pilot.json"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from . import (ExperimentConfig, Setting, analyze_file, exact_reference, generate_instances,
               plot_scaling, run_experiment, scaling_table, summary_table)


def main():
    parser = argparse.ArgumentParser(description="E1 · Блага и обязанности")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/pilot.json")
    run.add_argument("--jobs", type=int, help="Максимальное число одновременных запросов к модели")
    run.add_argument("--algorithms-only", action="store_true", help="Выполнить прогон без запросов к модели")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("path")
    analyze.add_argument("--allow-partial", action="store_true")
    scaling = sub.add_parser("scaling", help="Построить графики по n и m из завершённого журнала")
    scaling.add_argument("path")
    scaling.add_argument("--group")
    scaling.add_argument("--output", type=Path)
    reference = sub.add_parser("reference")
    reference.add_argument("--agents", type=int, default=3)
    reference.add_argument("--items", type=int, default=6)
    reference.add_argument("--kind", choices=["goods", "chores"], default="chores")
    args = parser.parse_args()
    if args.command == "reference":
        config = ExperimentConfig(settings=(Setting(args.agents, args.items),), n_base=1, groups=("uniform",))
        instance = generate_instances(config)[0]
        print(json.dumps({**instance.record(), **exact_reference(instance.weights, args.kind, config.reference).record()},
                         ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        config = ExperimentConfig.load(args.config)
        if args.jobs is not None:
            try:
                config = replace(config, jobs=args.jobs)
            except ValueError as exc:
                run.error(str(exc))
        if args.algorithms_only:
            config = replace(config, models=())

        def progress(position, total, _key):
            if position % 100 == 0 or position == total:
                print(f"Завершено шагов: {position}/{total}", flush=True)

        result = run_experiment(config, progress=progress if config.models else None)
    elif args.command == "scaling":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = analyze_file(args.path)
        if args.group and args.group not in result.config.groups:
            scaling.error(f"Неизвестная группа: {args.group}")
        stats = scaling_table(result, group=args.group)
        figures = plot_scaling(result, group=args.group, stats=stats)
        figures.update(plot_scaling(result, group=args.group, paper=True, stats=stats))
        if not figures:
            scaling.error("Нужны хотя бы два размера с одинаковым n или m")
        output = args.output or (result.directory / f"scaling_{args.group or 'all'}")
        output.mkdir(parents=True, exist_ok=True)
        stats.to_csv(output / "scaling_summary.csv", index=False)
        for name, figure in figures.items():
            figure.savefig(output / f"{name}.png", dpi=160)
            plt.close(figure)
        print(f"Графики и таблица: {output.resolve()}")
        return
    else:
        result = analyze_file(args.path, allow_partial=args.allow_partial)
    print(summary_table(result.rows).round(4).to_string())
    print(f"\nАртефакты: {result.directory.resolve()}; завершён: {result.complete}")


if __name__ == "__main__":
    main()
