"""Package the dashboard and aggregate experiment results for GitHub Pages."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
PUBLISHED = ROOT / "runs" / "published"
PILOT = PUBLISHED / "3x6_5matrices_gpt-6.1-sol_4prompts"
SCALING = PUBLISHED / "10sizes-3x6-to-10x50_1000matrices_algorithms"
NO_EFX = PUBLISHED / "3sizes-4x13-to-10x31_3matrices_no-efx_algorithms"
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
    matrix_ids = defaultdict(set)
    for row in read_csv(folder / "results.csv"):
        matrix_ids[row["group"]].add(row["base_id"])
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
        matrix_count = sum(map(len, matrix_ids.values())) if group == "all" else len(matrix_ids[group])
        groups[group] = {"label": group_label, "matrix_count": matrix_count, **domains}
    setting = sizes.pop()
    return {"setting": setting, "groups": groups, "prompt_count": len(config["prompt_variants"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    if SITE.is_relative_to(output):
        parser.error("--output must be a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("index.html", "app.js", "style.css"):
        shutil.copyfile(SITE / filename, output / filename)
    data = dataset(PILOT)
    downloads = output / "data"
    downloads.mkdir(parents=True, exist_ok=True)
    for filename in ("results.csv", "summary.csv", "radar_summary.csv", "bootstrap.csv"):
        shutil.copyfile(PILOT / filename, downloads / filename)
    for index, kind in enumerate(("goods", "chores"), 1):
        for scheme in ("paper", "extended"):
            shutil.copyfile(PILOT / f"{scheme}_radar_{index}.pdf", downloads / f"{scheme}-{kind}.pdf")
    figures = output / "scaling"
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCALING / "scaling_summary.csv", figures / "scaling_summary.csv")
    shutil.copyfile(NO_EFX / "no_efx_chores.csv", figures / "no_efx_chores.csv")
    for kind in ("goods", "chores"):
        for variable in ("n", "m", "paper"):
            filename = f"scaling_{variable}_{kind}.png"
            shutil.copyfile(SCALING / filename, figures / filename)
    payload = (json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    script = (output / "app.js").read_bytes()
    revision = hashlib.sha256(payload + script).hexdigest()[:12]
    data_file = f"data-{revision}.json"
    script_file = f"app-{revision}.js"
    (output / "data.json").write_bytes(payload)
    (output / data_file).write_bytes(payload)
    (output / script_file).write_bytes(script)
    page = output / "index.html"
    html = page.read_text(encoding="utf-8").replace('src="app.js"', f'src="{script_file}"', 1)
    html = html.replace('data-results="data.json"', f'data-results="{data_file}"', 1)
    page.write_text(html, encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"GitHub Pages dashboard built at {output}")


if __name__ == "__main__":
    main()
