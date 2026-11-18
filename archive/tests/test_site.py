import unittest

from archive.scripts.build_site import GROUPS, LATEST, dataset


class DashboardTests(unittest.TestCase):
    def test_input_group_labels_are_not_replaced_by_method_labels(self):
        groups = dataset(LATEST)["groups"]
        labels = {key: group["label"] for key, group in groups.items()}
        self.assertEqual(labels, {"all": "Все типы входов", **GROUPS})
