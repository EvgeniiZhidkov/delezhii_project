"""Independent small exhaustive checks for the newly reported science metrics."""
import itertools
import tempfile
import unittest

import numpy as np

from chores_e1 import ExperimentConfig, ReferenceConfig, analyze_file, run_experiment
from chores_e1.evaluation import build_evaluation, approximation_scores
from chores_e1.metrics import answer_metrics
from chores_e1.reference import round_robin, maximum_nash_welfare, minimum_social_welfare


def fraction(a, b):
    return min(1, a/b) if b > 0 else 1


class EvaluationTests(unittest.TestCase):
    def test_all_new_scores_against_literal_partitions(self):
        weights = np.array([[0, 1, 4, 4], [3, 1, 0, 2]], dtype=float)
        weights /= weights.sum(axis=1, keepdims=True)
        assignments = list(itertools.product(range(2), repeat=4))
        for kind in ("goods", "chores"):
            context = build_evaluation(weights, kind, ReferenceConfig(chunk_size=3))
            utilities = [[sum(weights[i,g] for g in range(4) if a[g] == i) for i in range(2)]
                         for a in assignments]
            mms = []
            for i in range(2):
                partitions = [[sum(weights[i,g] for g in range(4) if a[g] == j) for j in range(2)]
                              for a in assignments]
                mms.append(max(min(p) for p in partitions) if kind == "goods" else min(max(p) for p in partitions))
            for a, own in zip(assignments, utilities):
                score = approximation_scores(weights, a, kind, context)
                ef, ef1, efx = [], [], []
                for i, j in itertools.permutations(range(2), 2):
                    value = lambda owner: sum(weights[i,g] for g in range(4) if a[g] == owner)
                    other = value(j)
                    removable = [g for g in range(4) if a[g] == (j if kind == "goods" else i) and weights[i,g] > 0]
                    ef.append(fraction(own[i], other) if kind == "goods" else fraction(other, own[i]))
                    removals = [fraction(own[i], max(0, other-weights[i,g])) if kind == "goods"
                                else fraction(other, max(0, own[i]-weights[i,g])) for g in removable]
                    ef1.append(max(removals, default=ef[-1]))
                    efx.append(min(removals, default=ef[-1]))
                expected_mms = min(fraction(u,b) if kind == "goods" else fraction(b,u) for u,b in zip(own,mms))
                expected_sw = (fraction(sum(own), max(map(sum, utilities))) if kind == "goods"
                               else fraction(min(map(sum, utilities)), sum(own)))
                dominated = any(all(v >= u-1e-9 for v,u in zip(candidate,own)) and
                                any(v > u+1e-9 for v,u in zip(candidate,own)) if kind == "goods" else
                                all(v <= u+1e-9 for v,u in zip(candidate,own)) and
                                any(v < u-1e-9 for v,u in zip(candidate,own)) for candidate in utilities)
                self.assertAlmostEqual(score["ef_alpha"], min(ef))
                self.assertAlmostEqual(score["ef1_alpha"], min(ef1))
                self.assertAlmostEqual(score["efx_alpha"], min(efx))
                self.assertAlmostEqual(score["mms_alpha"], expected_mms)
                self.assertAlmostEqual(score["sw_ratio"], expected_sw)
                self.assertEqual(score["po"], not dominated)
                self.assertGreaterEqual(score["po_alpha"] + 1e-9, score["sw_ratio"])
                self.assertLessEqual(score["ef_alpha"], score["efx_alpha"] + 1e-9)
                self.assertLessEqual(score["efx_alpha"], score["ef1_alpha"] + 1e-9)
                if kind == "goods":
                    expected_nw = fraction(np.prod(own)**0.5, max(np.prod(u)**0.5 for u in utilities))
                    self.assertAlmostEqual(score["nw_ratio"], expected_nw)
                else:
                    self.assertIsNone(score["nw_ratio"])

    def test_round_robin_both_domains_and_min_sw_optimum(self):
        rng = np.random.default_rng(47)
        for n, m in [(2, 3), (3, 4), (3, 6)]:
            for _ in range(12):
                weights = rng.integers(1, 12, size=(n,m))
                for kind in ("goods", "chores"):
                    scores = answer_metrics(weights, round_robin(weights, kind), kind)
                    self.assertTrue(scores["ef1"])
                minsw = answer_metrics(weights, minimum_social_welfare(weights), "chores")
                self.assertAlmostEqual(minsw["sw_ratio"], 1)
                self.assertTrue(minsw["po"])

    def test_mnw_optimality_and_undefined_edge_cases(self):
        weights = [[9, 1, 4, 5], [1, 9, 5, 4]]
        answer = maximum_nash_welfare(weights)
        score = answer_metrics(weights, answer.owners, "goods")
        self.assertTrue(score["ef1"])
        self.assertTrue(score["po"])
        self.assertAlmostEqual(score["nw_ratio"], 1)
        self.assertIsNone(maximum_nash_welfare([[1], [1]]).owners)
        self.assertIsNone(maximum_nash_welfare(weights, ReferenceConfig(max_allocations=1)).owners)
        invalid = answer_metrics(weights, None, "goods")
        self.assertEqual(invalid["ef1_alpha"], 0)
        self.assertFalse(invalid["efx"])
        self.assertIsNone(answer_metrics(weights, None, "chores")["nw_ratio"])

    def test_nash_bound_does_not_overstate_mnw_ratio(self):
        rng = np.random.default_rng(46)
        for _ in range(6):
            weights = rng.integers(1, 12, size=(3, 6))
            exact = build_evaluation(weights, "goods")
            bounded = build_evaluation(weights, "goods", ReferenceConfig(max_allocations=1))
            self.assertGreaterEqual(bounded.nw_upper + 1e-12, exact.best_nw)
            owners = round_robin(weights)
            scores = approximation_scores(weights, owners, "goods", bounded)
            expected = approximation_scores(weights, owners, "goods", exact)
            self.assertGreater(scores["nw_ratio_lower"], 0)
            self.assertLessEqual(scores["nw_ratio_lower"], expected["nw_ratio"] + 1e-12)
            self.assertIsNone(scores["nw_ratio"])
            self.assertEqual(expected["nw_ratio_lower"], expected["nw_ratio"])
            self.assertIsNone(answer_metrics(weights, owners, "chores")["nw_ratio_lower"])

    def test_efx_ignores_zero_items_and_po_detects_single_agent_improvement(self):
        # One zero-valued good must not make EFX coincide with EF.
        score = answer_metrics([[0, 1, 2], [1, 1, 1]], [1, 0, 1], "goods")
        self.assertTrue(score["efx"])
        # Common-factor PO approximation can be 1 even if only one agent improves.
        score = answer_metrics([[1, 0], [1, 1]], [0, 0], "goods")
        self.assertFalse(score["po"])
        self.assertEqual(score["po_alpha"], 1)

    def test_saved_schema_two_is_replayable_and_named_correctly(self):
        from chores_e1.experiment import compare_saved_run
        from pathlib import Path
        runs = Path(__file__).resolve().parents[2] / "runs"
        path = runs / "legacy_3x6_25matrices_gpt-6.1-sol"
        if not (path / "journal.jsonl").is_file():
            path = runs / "model_gpt-6.1-sol"
        if not (path / "journal.jsonl").is_file():
            self.skipTest("The historical schema-2 pilot is archived locally in runs/")
        saved = analyze_file(path)
        self.assertNotIn("Эвристика", set(saved.rows["method"]))
        self.assertEqual(set(saved.rows[saved.rows["source"] == "heuristic"]["method"]), {"Round-Robin", "MinSW"})
        with tempfile.TemporaryDirectory() as temporary:
            comparison = compare_saved_run(path, output_dir=temporary)
            self.assertEqual(len(comparison.rows[comparison.rows["source"] == "model"]), 50)
            self.assertIn("MNW (exact)", set(comparison.rows["method"]))
            self.assertEqual(set(comparison.rows[comparison.rows["kind"] == "chores"]["method"]),
                             {"EF1-opt (exact)", "Round-Robin", "MinSW", "Codex"})
            self.assertTrue((comparison.directory / "comparison_manifest.json").exists())

    def test_radars_keep_the_full_circle_and_missing_chores_nash(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from chores_e1 import plot_results, radar_table, EXTENDED_METRICS
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(ExperimentConfig(n_base=1, groups=("uniform",), output_dir=temporary))
            stats = radar_table(result)
            self.assertTrue(stats[(stats["kind"] == "chores") & (stats["metric"] == "nw_ratio")]["mean"].isna().all())
            figures = plot_results(result, metrics=EXTENDED_METRICS)
            for figure in figures:
                figure.canvas.draw()
                for axis in figure.axes:
                    if axis.get_visible():
                        self.assertAlmostEqual(axis.get_xlim()[1] - axis.get_xlim()[0], 2*np.pi)
                        self.assertEqual(len(axis.get_xticklabels()), len(EXTENDED_METRICS))
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
