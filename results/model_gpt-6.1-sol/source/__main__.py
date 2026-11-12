"""python -m chores_e1 run --config configs/pilot.json"""
import argparse
import json
from . import ExperimentConfig, Setting, analyze_file, exact_reference, generate_instances, run_experiment, summary_table


def main():
    parser = argparse.ArgumentParser(description="E1 · Блага и обязанности")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/pilot.json")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("path")
    analyze.add_argument("--allow-partial", action="store_true")
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
        result = run_experiment(ExperimentConfig.load(args.config))
    else:
        result = analyze_file(args.path, allow_partial=args.allow_partial)
    print(summary_table(result.rows).round(4).to_string())
    print(f"\nАртефакты: {result.directory.resolve()}; завершён: {result.complete}")


if __name__ == "__main__":
    main()
