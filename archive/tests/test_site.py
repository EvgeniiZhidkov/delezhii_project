import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from archive.scripts.build_site import GROUPS, LATEST, dataset, main


class DashboardTests(unittest.TestCase):
    def test_input_group_labels_are_not_replaced_by_method_labels(self):
        groups = dataset(LATEST)["groups"]
        labels = {key: group["label"] for key, group in groups.items()}
        self.assertEqual(labels, {"all": "Все типы входов", **GROUPS})

    def test_page_loads_assets_versioned_by_content(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("sys.argv", ["build_site.py", "--output", str(output)]):
                main()

            script = (output / "app.js").read_bytes()
            payload = (output / "data.json").read_bytes()
            revision = hashlib.sha256(payload + script).hexdigest()[:12]
            html = (output / "index.html").read_text(encoding="utf-8")

            self.assertIn(f'src="app-{revision}.js" data-results="data-{revision}.json"', html)
            self.assertEqual((output / f"app-{revision}.js").read_bytes(), script)
            self.assertEqual((output / f"data-{revision}.json").read_bytes(), payload)
