"""Rebuild paper-style figures from saved replies, with no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chores_e1 import compare_saved_run, plot_results, radar_table, summary_table, summarize
from chores_e1.plotting import EXTENDED_METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/revised_figures")
    args = parser.parse_args()
    for name in ("model_gpt-6.1-sol", "algorithms_3x6_n20"):
        source = ROOT / "results" / name
        result = compare_saved_run(source, output_dir=ROOT / "runs")
        output = args.output / name
        output.mkdir(parents=True, exist_ok=True)
        result.rows.to_csv(output / "results.csv", index=False)
        summary_table(result.rows).to_csv(output / "summary.csv")
        radar_table(result).to_csv(output / "radar_summary.csv", index=False)
        summarize(result.rows, n_base=result.config.n_base, seed=result.config.seed+1).to_csv(
            output / "bootstrap.csv", index=False)
        for prefix, metrics in (("paper", None), ("extended", EXTENDED_METRICS)):
            figures = plot_results(result, **({"metrics": metrics} if metrics else {}))
            for index, figure in enumerate(figures):
                setting = result.config.settings[index // 2].label
                kind = ("goods", "chores")[index % 2]
                for extension in ("png", "pdf", "svg"):
                    figure.savefig(output / f"{prefix}_{setting}_{kind}.{extension}", dpi=180, bbox_inches="tight")
                plt.close(figure)
        provenance = {"source_journal": str(source.relative_to(ROOT) / "journal.jsonl"),
                      "source_sha256": hashlib.sha256((source / "journal.jsonl").read_bytes()).hexdigest(),
                      "model_calls": 0, "config": result.config.record(),
                      "source_sha256_current": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in sorted((ROOT / "chores_e1").glob("*.py"))}}
        (output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(f"{name}: {len(result.rows)} allocations, figures saved to {output}")


if __name__ == "__main__":
    main()
