"""Rebuild tables and figures for the published experiment journal."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chores_e1 import analyze_file, plot_results, radar_table, summary_table, summarize
from chores_e1.plotting import EXTENDED_METRICS, PAPER_METRICS


def main():
    source = ROOT / "results" / "latest"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=source)
    args = parser.parse_args()
    result = analyze_file(source)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(output / "results.csv", index=False)
    result.references.drop(columns=["weights"]).to_csv(output / "references.csv", index=False)
    summary_table(result.rows).to_csv(output / "summary.csv")
    radar_table(result).to_csv(output / "radar_summary.csv", index=False)
    summarize(result.rows, n_base=result.config.n_base, seed=result.config.seed + 1).to_csv(
        output / "bootstrap.csv", index=False)
    for prefix, metrics in (("paper", PAPER_METRICS), ("extended", EXTENDED_METRICS)):
        for index, figure in enumerate(plot_results(result, metrics=metrics), 1):
            for extension in ("png", "pdf", "svg"):
                figure.savefig(output / f"{prefix}_radar_{index}.{extension}", dpi=180, bbox_inches="tight")
            plt.close(figure)
    print(f"{len(result.rows)} allocations, figures saved to {output}")


if __name__ == "__main__":
    main()
