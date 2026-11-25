"""Export the matrix identifiers, method labels and scores used by the site."""
import argparse
import csv
from pathlib import Path


RESULT_COLUMNS = (
    "setting", "n_agents", "n_items", "base_id", "group", "kind", "source", "method",
    "repeat", "prompt_variant", "valid", "ef1", "d", "envy", "total", "gap",
    "sd_ef", "sd_ef1", "sd_envy", "sd_d", "sd_ef_pairs", "sd_ef1_pairs",
    "wrong_side_only", "goods_rule_only", "prop", "ef", "efx", "mms", "po", "mnw", "prop1",
    "prop_alpha", "ef_alpha", "ef1_alpha", "mms_alpha", "po_alpha", "sw_ratio",
    "nw_ratio", "nw_ratio_lower", "efx_alpha", "prop1_alpha",
)


def export_results(folder):
    target = folder / "public_results.csv"
    with (folder / "results.csv").open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        columns = [column for column in RESULT_COLUMNS if column in (rows.fieldnames or ())]
        required = {"setting", "base_id", "group", "kind", "method"}
        if not required <= set(columns):
            raise ValueError(f"В {source.name} нет обязательных столбцов")
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Папки прогонов с results.csv")
    args = parser.parse_args()
    for folder in args.runs:
        print(export_results(folder))


if __name__ == "__main__":
    main()
