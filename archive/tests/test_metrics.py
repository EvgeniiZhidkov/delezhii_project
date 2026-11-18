"""Compare production vectorization with explicit definitions, including removals."""
import itertools
import unittest

import numpy as np

from chores_e1 import allocation_scores, sd_scores


def literal_cardinal(weights, owners, kind):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum(axis=1, keepdims=True)
    bundles = [[g for g, owner in enumerate(owners) if owner == j] for j in range(len(weights))]
    envy, residual = 0, 0
    for i, j in itertools.permutations(range(len(weights)), 2):
        own, other = bundles[i], bundles[j]
        value = lambda bundle: sum(weights[i, g] for g in bundle)
        delta = value(other) - value(own) if kind == "goods" else value(own) - value(other)
        envy = max(envy, delta)
        removable = other if kind == "goods" else own
        after = []
        for g in removable:
            after.append(value([h for h in other if h != g]) - value(own) if kind == "goods"
                         else value([h for h in own if h != g]) - value(other))
        residual = max(residual, max(0, min(after)) if after else 0)
    return envy, residual


def literal_sd(weights, owners, kind):
    bundles = [[g for g, owner in enumerate(owners) if owner == j] for j in range(len(weights))]
    passes, passes_one = [], []
    for i, j in itertools.permutations(range(len(weights)), 2):
        def dominates(own, other):
            for q in set(weights[i]):
                if q == 0:
                    continue
                a = sum(weights[i, g] >= q for g in own)
                b = sum(weights[i, g] >= q for g in other)
                if (a < b if kind == "goods" else a > b):
                    return False
            return True
        own, other = bundles[i], bundles[j]
        passes.append(dominates(own, other))
        after = [dominates(own, [h for h in other if h != g]) for g in other] if kind == "goods" else [
            dominates([h for h in own if h != g], other) for g in own]
        passes_one.append(passes[-1] or any(after))
    return all(passes), all(passes_one)


class MetricTests(unittest.TestCase):
    def test_all_allocations_match_literal_definitions(self):
        weights = np.array([[0, 1, 4, 4], [3, 1, 2, 2], [1, 5, 0, 3]])
        owners = np.array(list(itertools.product(range(3), repeat=4)))
        for kind in ("goods", "chores"):
            cardinal = allocation_scores(weights, owners, kind)
            sd = sd_scores(weights, owners, kind)
            for k, assignment in enumerate(owners):
                envy, d = literal_cardinal(weights, assignment, kind)
                self.assertAlmostEqual(cardinal["d"][k], d)
                self.assertAlmostEqual(cardinal["envy"][k], envy)
                expected_ef, expected_ef1 = literal_sd(weights, assignment, kind)
                self.assertEqual(sd["sd_ef"][k], expected_ef)
                self.assertEqual(sd["sd_ef1"][k], expected_ef1)
                self.assertEqual(sd["sd_d"][k] == 0, expected_ef1)
                self.assertEqual(sd["sd_envy"][k] == 0, expected_ef)
                if expected_ef1:
                    self.assertLessEqual(cardinal["d"][k], 1e-9)

    def test_ef1_does_not_imply_sd_ef1(self):
        for kind, row, owners in [
            ("goods", [3, 3, 4, 4, 1], [0, 0, 1, 1, 1]),
            ("chores", [4, 4, 1, 3, 3], [0, 0, 0, 1, 1]),
        ]:
            weights = np.array([row, row])
            self.assertLessEqual(allocation_scores(weights, owners, kind)["d"][0], 1e-9)
            self.assertFalse(sd_scores(weights, owners, kind)["sd_ef1"][0])

    def test_ties_and_zero_items(self):
        weights = np.array([[0, 1, 1, 3], [2, 0, 2, 4]])
        owners = np.array([0, 0, 1, 1])
        permutation = [0, 2, 1, 3]
        for kind in ("goods", "chores"):
            first, second = sd_scores(weights, owners, kind), sd_scores(weights[:, permutation], owners[permutation], kind)
            for metric in first:
                np.testing.assert_equal(first[metric], second[metric])
        zero_to_i = np.array([1, 0])
        # Participant 0's only extra item is neutral: no SD envy from participant 1 either.
        neutral = np.array([[1, 0], [1, 0]])
        self.assertTrue(sd_scores(neutral, zero_to_i, "goods")["sd_ef1"][0])

    def test_single_agent_and_validation(self):
        self.assertTrue(sd_scores([[1, 2]], [0, 0], "chores")["sd_ef"][0])
        self.assertEqual(allocation_scores([[1, 2]], [0, 0], "goods")["d"][0], 0)
        for weights in ([[0, 0], [1, 2]], [[-1, 2]], [[float("nan"), 2]]):
            with self.assertRaises(ValueError): allocation_scores(weights, [0, 0], "goods")
        for owners in ([0, 2], [0.0, 1.0], [0], [False, True]):
            with self.assertRaises(ValueError): allocation_scores([[1, 2], [2, 1]], owners, "goods")
        with self.assertRaises(ValueError): allocation_scores([[1]], [0], "bad")


if __name__ == "__main__":
    unittest.main()
