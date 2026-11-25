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

    def test_fallback_contains_interactive_and_report_runs(self):
        files = json.loads((self.output / "data.json").read_bytes())["files"]
        filenames = {Path(path).name for path in files}
        self.assertTrue({"results.csv", "scaling_summary.csv", "no_efx_chores.csv"} <= filenames)
        for path in files:
            self.assertTrue(path.startswith("runs/published/"))
            self.assertEqual((self.output / path).read_bytes(), (ROOT / path).read_bytes())

    def test_page_loads_assets_versioned_by_content(self):
        script = (self.output / "app.js").read_bytes()
        payload = (self.output / "data.json").read_bytes()
        revision = hashlib.sha256(payload + script).hexdigest()[:12]
        html = (self.output / "index.html").read_text(encoding="utf-8")

        self.assertIn(f'src="app-{revision}.js" data-results="data-{revision}.json"', html)
        self.assertEqual((self.output / f"app-{revision}.js").read_bytes(), script)
        self.assertEqual((self.output / f"data-{revision}.json").read_bytes(), payload)

    def test_fallback_excludes_journals_and_source_snapshots(self):
        self.assertFalse(list(self.output.rglob("journal.jsonl")))
        self.assertFalse(list(self.output.rglob("source/")))

    def test_builder_is_excluded_from_published_files(self):
        self.assertFalse((self.output / "build_site.py").exists())
