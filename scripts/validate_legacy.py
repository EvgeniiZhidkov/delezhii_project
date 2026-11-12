"""Compare all original pilot allocations; only offline cells of the archive execute."""
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from chores_e1 import ExperimentConfig, allocation_scores, exact_reference, generate_instances


def main():
    original = json.loads((ROOT / "archive" / "chores_e1_3_original.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    # Imports/settings, scorer, parser/prompt, and baseline assertions. The model
    # cell (12) is deliberately not executed.
    with contextlib.redirect_stdout(io.StringIO()):
        for i in (2, 6, 8, 10):
            exec(compile("".join(original["cells"][i]["source"]), f"original-cell-{i}", "exec"), namespace)
    compared = 0
    for instance in generate_instances(ExperimentConfig(n_base=20)):
        old = namespace["instances"][instance.base_id, instance.group]
        np.testing.assert_equal(instance.weights, old)
        for kind in ("goods", "chores"):
            old_scores = namespace["allocation_scores"](old, namespace["ALL_OWNERS"], kind)
            new_scores = allocation_scores(instance.weights, namespace["ALL_OWNERS"], kind)
            for key in old_scores:
                np.testing.assert_allclose(new_scores[key], old_scores[key], atol=1e-12, rtol=0)
            ref = exact_reference(instance.weights, kind)
            old_ref = namespace["exact_reference"](old, kind)
            np.testing.assert_equal(ref.owners, old_ref[0])
            np.testing.assert_allclose([ref.best_d, ref.best_total], old_ref[1:], atol=1e-10, rtol=0)
            compared += len(namespace["ALL_OWNERS"])
    print(f"Regression passed: {compared} allocations, 100 inputs, both goods and chores.")


if __name__ == "__main__":
    main()
