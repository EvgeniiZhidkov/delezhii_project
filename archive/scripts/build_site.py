"""Package the dashboard and aggregate experiment results for GitHub Pages."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "results" / "latest"
GROUPS = {
    "paper": "Общая шкала (paper)",
    "uniform": "Независимые оценки (uniform)",
    "r=2": "Два уровня (1 и 2)",
    "r=5": "Два уровня (1 и 5)",
    "r=10": "Два уровня (1 и 10)",
}
METRICS = (
    "prop_alpha", "ef_alpha", "ef1_alpha", "po_alpha", "sw_ratio", "mms_alpha",
    "efx_alpha", "sd_ef1", "nw_ratio", "ef1", "efx", "po", "total",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(rows, radar):
    """Use the published radar means for the overall view, grouped means otherwise."""
    instances = sum(int(row["n_instances"]) for row in rows)
    means = {}
    for key in METRICS:
        known = [(float(row[key]), int(row["n_instances"])) for row in rows if row[key]]
        denominator = sum(n for _, n in known)
        means[key] = (sum(value * n for value, n in known) / denominator if denominator else None)
    means.update(radar)
    return {"instances": instances, "values": means}


def dataset(folder):
    config = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["config"]
    model_names = {model["label"]: model["model"] or model["label"] for model in config["models"]}
    summary = read_csv(folder / "summary.csv")
    sizes = {row["setting"] for row in summary}
    if len(sizes) != 1:
        raise ValueError(f"Expected one matrix size in {folder / 'summary.csv'}")
    radars = defaultdict(dict)
    for row in read_csv(folder / "radar_summary.csv"):
        radars[(row["kind"], row["method"])][row["metric"]] = float(row["mean"]) if row["mean"] else None
    conditions = {"all": "Все типы входов", **{key: GROUPS.get(key, key) for key in config["groups"]}}
    groups = {}
    for group, group_label in conditions.items():
        domains = {"goods": [], "chores": []}
        grouped = defaultdict(list)
        for row in summary:
            if group == "all" or row["group"] == group:
                grouped[(row["kind"], row["method"])].append(row)
        for (kind, method), rows in grouped.items():
            model_label, separator, variant = method.partition(" · ")
            is_model = model_label in model_names
            method_label = model_names[model_label] + separator + variant if is_model else method
            domains[kind].append({
                "method": method,
                "label": method_label,
                "is_model": is_model,
                **aggregate(rows, radars[(kind, method)] if group == "all" else {}),
            })
        groups[group] = {"label": group_label, **domains}
    setting = sizes.pop()
    return {"setting": setting, "groups": groups, "prompt_count": len(config["prompt_variants"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or (ROOT / "site").is_relative_to(output):
        parser.error("--output must be a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "site", output, dirs_exist_ok=True)
    data = dataset(LATEST)
    downloads = output / "data"
    downloads.mkdir(parents=True, exist_ok=True)
    for filename in ("results.csv", "summary.csv", "radar_summary.csv", "bootstrap.csv"):
        shutil.copyfile(LATEST / filename, downloads / filename)
    for index, kind in enumerate(("goods", "chores"), 1):
        for scheme in ("paper", "extended"):
            shutil.copyfile(LATEST / f"{scheme}_radar_{index}.pdf", downloads / f"{scheme}-{kind}.pdf")
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"GitHub Pages dashboard built at {output}")


if __name__ == "__main__":
    main()
