import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.output = Path(temporary.name)
        subprocess.run(
            [sys.executable, str(ROOT / "site" / "build_site.py"), "--output", str(cls.output)],
            cwd=ROOT,
            check=True,
        )

    def test_input_group_labels_are_not_replaced_by_method_labels(self):
        groups = json.loads((self.output / "data.json").read_bytes())["groups"]
        labels = {key: group["label"] for key, group in groups.items()}
        self.assertEqual(labels, {
            "all": "Все типы входов",
            "paper": "Общая шкала (paper)",
            "uniform": "Независимые оценки (uniform)",
            "r=2": "Два уровня (1 и 2)",
            "r=5": "Два уровня (1 и 5)",
            "r=10": "Два уровня (1 и 10)",
        })

    def test_page_loads_assets_versioned_by_content(self):
        script = (self.output / "app.js").read_bytes()
        payload = (self.output / "data.json").read_bytes()
        revision = hashlib.sha256(payload + script).hexdigest()[:12]
        html = (self.output / "index.html").read_text(encoding="utf-8")

        self.assertIn(f'src="app-{revision}.js" data-results="data-{revision}.json"', html)
        self.assertEqual((self.output / f"app-{revision}.js").read_bytes(), script)
        self.assertEqual((self.output / f"data-{revision}.json").read_bytes(), payload)

    def test_matrix_counts_use_inputs_shared_by_methods_and_modes(self):
        groups = json.loads((self.output / "data.json").read_bytes())["groups"]
        self.assertEqual({key: group["matrix_count"] for key, group in groups.items()}, {
            "all": 5,
            "paper": 1,
            "uniform": 1,
            "r=2": 1,
            "r=5": 1,
            "r=10": 1,
        })

    def test_builder_is_excluded_from_published_files(self):
        self.assertFalse((self.output / "build_site.py").exists())
