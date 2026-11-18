"""Package the dashboard and aggregate experiment results for GitHub Pages."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "results" / "revised_figures"
DATASETS = {
    "pilot": ("model_gpt-6.1-sol", "Пилот с gpt-6.1-sol"),
    "algorithms": ("algorithms_3x6_n20", "Алгоритмы на 100 матрицах"),
}
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
METHODS = ("Round-Robin", "MaxSW", "MinSW", "MNW (exact)", "Codex", "EF1-opt (exact)")


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


def dataset(folder, title):
    summary = read_csv(folder / "summary.csv")
    sizes = {row["setting"] for row in summary}
    if len(sizes) != 1:
        raise ValueError(f"Expected one matrix size in {folder / 'summary.csv'}")
    radars = defaultdict(dict)
    for row in read_csv(folder / "radar_summary.csv"):
        radars[(row["kind"], row["method"])][row["metric"]] = float(row["mean"]) if row["mean"] else None
    conditions = {"all": "Все типы входов", **GROUPS}
    groups = {}
    for group, label in conditions.items():
        domains = {}
        grouped = defaultdict(list)
        for row in summary:
            if group == "all" or row["group"] == group:
                grouped[(row["kind"], row["method"])].append(row)
        for kind in ("goods", "chores"):
            domains[kind] = [
                {"method": method, "label": "gpt-6.1-sol" if method == "Codex" else method,
                 **aggregate(grouped[(kind, method)], radars[(kind, method)] if group == "all" else {})}
                for method in METHODS if (kind, method) in grouped
            ]
        groups[group] = {"label": label, **domains}
    setting = sizes.pop()
    return {"title": title, "setting": setting, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or (ROOT / "site").is_relative_to(output):
        parser.error("--output must be a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "site", output, dirs_exist_ok=True)
    data = {}
    for key, (folder, title) in DATASETS.items():
        source = REPORTS / folder
        data[key] = dataset(source, title)
        downloads = output / "data" / key
        downloads.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.csv", "radar_summary.csv", "bootstrap.csv"):
            shutil.copyfile(source / filename, downloads / filename)
        for kind in ("goods", "chores"):
            for scheme in ("paper", "extended"):
                name = f"{scheme}_{data[key]['setting']}_{kind}.pdf"
                shutil.copyfile(source / name, downloads / f"{scheme}-{kind}.pdf")
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"GitHub Pages dashboard built at {output}")


if __name__ == "__main__":
    main()
