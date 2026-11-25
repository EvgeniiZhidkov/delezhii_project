from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np
import pandas as pd

from chores_e1 import (ExperimentConfig, ModelConfig, ReferenceConfig, Setting, analyze_file,
                       allocation_scores, answer_metrics, exact_reference, generate_instances, paired_differences,
                       make_prompt, parse_allocation, run_experiment, summarize, summary_table)
from chores_e1.backends import BackendReply, inspect_events
from chores_e1.metrics import METRICS


class ReferenceTests(unittest.TestCase):
    def test_chunking_matches_exhaustive_oracle(self):
        from archive.tests.test_metrics import literal_cardinal
        rng = np.random.default_rng(8)
        for _ in range(5):
            weights = rng.integers(0, 8, size=(3, 4))
            for kind in ("goods", "chores"):
                candidates = []
                for owners in itertools.product(range(3), repeat=4):
                    _, d = literal_cardinal(weights, owners, kind)
                    total = 100 * sum(weights[i, g] / weights[i].sum() for g, i in enumerate(owners))
                    candidates.append((d, -total if kind == "goods" else total))
                expected_d = min(d for d, _ in candidates)
                expected_total = min(t for d, t in candidates if d <= expected_d + 1e-9)
                result = exact_reference(weights, kind, ReferenceConfig(chunk_size=7))
                self.assertTrue(result.certified, result.message)
                self.assertAlmostEqual(result.best_d, expected_d)
                self.assertAlmostEqual(result.best_total, -expected_total if kind == "goods" else expected_total, places=6)

    def test_large_reference_never_enumerates_without_budget(self):
        result = exact_reference(np.ones((3, 7)), "chores", ReferenceConfig(max_allocations=1000))
        self.assertFalse(result.certified)
        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.owners)

    def test_skipped_reference_does_not_define_efficiency_gap(self):
        from chores_e1.reference import ReferenceResult
        ref = ReferenceResult("enumerate", "skipped")
        metrics = answer_metrics([[1, 1], [1, 1]], [0, 1], "goods", ref)
        self.assertIsNone(metrics["gap"])


class DataAndParserTests(unittest.TestCase):
    def test_old_pilot_matrix_and_nested_variants(self):
        config = ExperimentConfig(n_base=2)
        inputs = generate_instances(config)
        np.testing.assert_equal(inputs[0].weights, [[10, 71, 133, 52, 29, 110], [8, 39, 78, 44, 19, 67], [10, 62, 116, 63, 13, 97]])
        bigger = generate_instances(replace(config, settings=(Setting(3, 5), Setting())))
        indexed = {x.key: x.weights for x in bigger}
        for instance in inputs:
            np.testing.assert_equal(instance.weights, indexed[instance.key])
        for base in range(2):
            matrices = [x.weights for x in inputs if x.base_id == base and x.group.startswith("r=")]
            for matrix in matrices[1:]:
                np.testing.assert_equal(matrix > 1, matrices[0] > 1)

    def test_scaling_profile_is_balanced_and_no_ef_is_certified(self):
        config = ExperimentConfig.load(Path(__file__).resolve().parents[2] / "configs" / "scaling.json")
        instances = generate_instances(config)
        self.assertEqual(len(instances), 1000)
        self.assertEqual(len({instance.key for instance in instances}), 1000)
        self.assertEqual(len(config.settings), 10)
        hard = [instance for instance in instances if instance.group == "no_ef"]
        self.assertEqual(len(hard), 200)
        for instance in hard:
            n, m = instance.weights.shape
            common = instance.weights // (3 * m + 1)
            np.testing.assert_equal(common, np.broadcast_to(common[0], (n, m)))
            self.assertNotEqual(common[0].sum() % n, 0)
            if instance.setting == Setting(3, 6):
                assignments = np.array(list(itertools.product(range(n), repeat=m)))
                for kind in ("goods", "chores"):
                    scores = allocation_scores(instance.weights, assignments, kind)
                    self.assertTrue((scores["envy"] > config.atol).all())
                    self.assertTrue((scores["d"] <= config.atol).any())

    def test_he_tao_case_has_no_efx_after_relabeling(self):
        config = ExperimentConfig.load(Path(__file__).resolve().parents[2] / "configs" / "no_efx.json")
        sample = generate_instances(config)
        self.assertEqual([instance.weights.shape for instance in sample], [(4, 13), (5, 18), (10, 31)])
        # Identical columns are interchangeable. Enumerate counts of each chore type per agent.
        costs, multiplicities = np.unique(sample[0].weights.T, axis=0, return_counts=True)
        self.assertEqual(sorted(multiplicities.tolist()), [3, 5, 5])
        self.assertEqual(np.unique(costs).tolist(), [1, 7, 20])
        splits = [[part for part in itertools.product(range(count + 1), repeat=4) if sum(part) == count]
                  for count in multiplicities]
        allocations = np.array(list(itertools.product(*splits))).transpose(0, 2, 1)
        bundle_costs = allocations @ costs
        self_cost = bundle_costs.diagonal(axis1=1, axis2=2)
        cheapest_own = np.where(allocations > 0, costs.T, np.inf).min(axis=2)
        self.assertFalse(np.any((self_cost - cheapest_own <= bundle_costs.min(axis=1)).all(axis=1)))

    def test_strict_parser(self):
        raw = '{"status":"allocation","bundles":{"A1":["G1"],"A2":["G2"]}}'
        np.testing.assert_equal(parse_allocation(raw, 2, 2), [0, 1])
        invalid = ["```json\n" + raw + "\n```", raw + " пояснение", raw.replace('"G2"', '"G1"'),
                   raw.replace('"G2"', '"G3"'), raw.replace('["G2"]', '[]'),
                   raw.replace('"status":"allocation"', '"status":"allocation","status":"allocation"'),
                   raw.replace('"G2"', 'NaN'), '[]']
        for candidate in invalid:
            with self.subTest(raw=candidate):
                with self.assertRaises(ValueError): parse_allocation(candidate, 2, 2)

    def test_event_audit(self):
        self.assertEqual(inspect_events('{"type":"turn.completed","usage":{"input_tokens":2}}'), {"input_tokens": 2})
        for stream in ['{"type":"turn.failed"}', '{"type":"turn.completed"}\n{"type":"turn.completed"}',
                       '{"type":"item.completed","item":{"type":"command_execution"}}\n{"type":"turn.completed"}']:
            with self.assertRaises(ValueError): inspect_events(stream)


class ReplayTests(unittest.TestCase):
    def test_parallel_model_calls_keep_sequential_replay_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(settings=(Setting(2, 2),), n_base=2, groups=("uniform",),
                                      models=(ModelConfig(),), jobs=2, output_dir=temporary)
            response = BackendReply(raw='{"status":"allocation","bundles":{"A1":["G1"],"A2":["G2"]}}')
            second_ready = Event()
            finished = []

            def backend(_prompt, _model, call_dir):
                position = int(call_dir.name)
                if position == 5:
                    self.assertTrue(second_ready.wait(5), "The next request must run concurrently")
                finished.append(position)
                if position == 9:
                    second_ready.set()
                return response

            result = run_experiment(config, backend=backend)
            self.assertEqual(finished[:2], [9, 5])
            sequential = run_experiment(replace(config, jobs=1), backend=lambda *_: response)
            replay = analyze_file(result.directory)
            self.assertEqual(replay.config.jobs, 2)
            pd.testing.assert_frame_equal(result.rows, replay.rows)
            pd.testing.assert_frame_equal(sequential.rows, result.rows)

    def test_prompt_variants_share_inputs_but_keep_separate_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            variants = ("free", "ef1", "prop", "welfare")
            config = ExperimentConfig(settings=(Setting(2, 2),), n_base=1, groups=("uniform",),
                                      repeats=2, models=(ModelConfig(),), prompt_variants=variants,
                                      output_dir=temporary)
            def backend(prompt, _model, _directory):
                bundles = (dict(A1=["G1", "G2"], A2=[]) if "считаешь справедливым" in prompt
                           else dict(A1=["G1"], A2=["G2"]))
                return BackendReply(raw=json.dumps(dict(status="allocation", bundles=bundles)), seconds=0.01)

            result = run_experiment(config, backend=backend)
            self.assertRegex(result.directory.name, r"^e1_2x2_1matrix_uniform_Codex_4prompts_\d{8}T\d{6}_[0-9a-f]{8}$")
            path = result.directory / "journal.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            model_trials = [r for r in records if r.get("source") == "model"]
            self.assertEqual(len(model_trials), 16)
            self.assertEqual(len({tuple(r["key"]) for r in model_trials}), 16)
            self.assertTrue(all(r["weights"] == model_trials[0]["weights"] for r in model_trials))
            for kind in ("goods", "chores"):
                prompts = {r["prompt_variant"]: r["prompt"] for r in model_trials if r["kind"] == kind}
                for prompt in prompts.values():
                    self.assertEqual(prompt.split("\n\n")[:2], prompts["free"].split("\n\n")[:2])
                    self.assertEqual(prompt.split("\n\n")[-1], prompts["free"].split("\n\n")[-1])
                self.assertEqual(prompts["free"].split("\n\n")[2], "Распредели предметы так, как считаешь справедливым.")
                expected_removal = "из X_j самое ценное" if kind == "goods" else "из X_i самую затратную"
                self.assertIn(expected_removal, prompts["ef1"])
            summary = summary_table(result.rows[result.rows["source"] == "model"])
            self.assertEqual(len(summary), 8)
            self.assertEqual(summary.loc[("2×2", "Codex · Без метрики", "goods", "uniform"), "ef1"], 0)
            self.assertEqual(summary.loc[("2×2", "Codex · EF1", "goods", "uniform"), "ef1"], 1)
            pd.testing.assert_frame_equal(result.rows, analyze_file(path).rows)
            model_trials[0]["prompt_variant"] = "ef1"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            with self.assertRaisesRegex(ValueError, "Вариант промпта"):
                analyze_file(path)

    def test_schema_three_replay_keeps_original_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(n_base=1, groups=("uniform",), models=(ModelConfig(),), output_dir=temporary)
            simulated = BackendReply(raw='{"status":"allocation","bundles":{"A1":["G1","G2"],"A2":["G3","G4"],"A3":["G5","G6"]}}')
            result = run_experiment(config, backend=lambda *_: simulated)
            path = result.directory / "journal.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0].update(schema=3, prompt_version="e1-paired-v1")
            records[0]["config"].pop("prompt_variants")
            for record in records:
                if record.get("source") == "model":
                    record["key"].pop()
                    record.pop("prompt_variant")
                    self.assertEqual(record["prompt"], make_prompt(record["weights"], record["kind"]))
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            pd.testing.assert_frame_equal(result.rows, analyze_file(path).rows)

    def test_replay_legacy_enumeration_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(n_base=1, groups=("uniform",), output_dir=temporary)
            result = run_experiment(config)
            path = result.directory / "journal.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["config"]["reference"].update(method="auto", time_limit=30.0)
            for record in records:
                if record.get("source") == "reference":
                    record["reference"].update(mip_gap=None, dual_bound=None)
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            pd.testing.assert_frame_equal(result.rows, analyze_file(path).rows)

    def test_skipped_reference_keeps_algorithms_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(n_base=1, groups=("uniform",), output_dir=temporary,
                                      reference=ReferenceConfig(max_allocations=1))
            result = run_experiment(config)
            self.assertEqual(result.references["status"].tolist(), ["skipped", "skipped"])
            self.assertEqual(result.rows["source"].tolist(), ["round_robin", "social_welfare"] * 2)
            self.assertTrue(result.rows["gap"].isna().all())
            self.assertTrue(result.rows[["mms_alpha", "po_alpha", "nw_ratio"]].isna().all().all())
            pd.testing.assert_frame_equal(result.rows, analyze_file(result.directory).rows)

    def test_run_replay_invalid_answer_and_prompt_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(settings=(Setting(2, 3),), n_base=2, groups=("uniform",),
                                      repeats=2, models=(ModelConfig(),), output_dir=temporary)
            def fake_backend(prompt, model, call_dir):
                return BackendReply(raw="это не JSON", seconds=0.01)
            result = run_experiment(config, backend=fake_backend)
            replay = analyze_file(result.directory)
            pd.testing.assert_frame_equal(result.rows, replay.rows)
            models = result.rows[result.rows["source"] == "model"]
            self.assertEqual(len(models), 8)
            self.assertEqual(models["ef1"].sum(), 0)
            self.assertEqual(models["sd_ef1"].sum(), 0)
            self.assertTrue(models["d"].isna().all())
            path = result.directory / "journal.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            next(record for record in records if record.get("source") == "model")["prompt"] += "tampered"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            with self.assertRaisesRegex(ValueError, "Промпт"):
                analyze_file(path)

    def test_incomplete_run_requires_explicit_diagnostic_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(n_base=1, groups=("uniform",), output_dir=temporary)
            result = run_experiment(config)
            path = result.directory / "journal.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "не завершён"): analyze_file(path)
            self.assertFalse(analyze_file(path, allow_partial=True).complete)

    def test_backend_error_is_preserved_before_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(n_base=1, groups=("uniform",), models=(ModelConfig(),), output_dir=temporary)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                run_experiment(config, backend=lambda *args: BackendReply(error="fixture failure"))
            path = next(Path(temporary).glob("*/journal.jsonl"))
            self.assertIn('"error": "fixture failure"', path.read_text())
            with self.assertRaises(ValueError): analyze_file(path)
            self.assertFalse(analyze_file(path, allow_partial=True).complete)


class AnalysisTests(unittest.TestCase):
    def test_repeat_weighting_paired_bootstrap_and_missing_metrics(self):
        rows = []
        for base, kind, values in [(0, "goods", [1, 1, 1]), (0, "chores", [0, 0, 0]),
                                   (1, "goods", [0]), (1, "chores", [1])]:
            for value in values:
                metrics = dict.fromkeys(METRICS, None)
                metrics.update(valid=True, ef1=value, sd_ef=value, sd_ef1=value)
                rows.append(dict(setting="2×3", method="fixture", kind=kind, group="uniform", base_id=base, **metrics))
        data = pd.DataFrame(rows)
        summary = summarize(data, n_base=2, n_boot=1000)
        ef1 = summary[summary["metric"] == "ef1"]
        self.assertTrue((ef1["mean"] == 0.5).all())
        difference = paired_differences(data, n_base=2, n_boot=1000).iloc[0]
        self.assertEqual(difference["mean"], 0)
        self.assertEqual(difference["n_instances"], 2)
        self.assertTrue(summary[summary["metric"] == "d"]["mean"].isna().all())


if __name__ == "__main__":
    unittest.main()
