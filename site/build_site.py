"""Package the dashboard and its fallback data for GitHub Pages."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
DATA_FILES = {"results.csv", "scaling_summary.csv", "no_efx_chores.csv"}
PUBLIC_FILES = DATA_FILES | {
    "manifest.json", "summary.csv", "radar_summary.csv", "bootstrap.csv",
    *(f"{scheme}_radar_{index}.pdf" for scheme in ("paper", "extended") for index in (1, 2)),
    *(f"scaling_{axis}_{kind}.png" for axis in ("paper", "n", "m") for kind in ("goods", "chores")),
}


def fallback_files(output):
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "runs/"], cwd=ROOT, text=True,
    )
    candidates = {
        path for path in map(Path, tracked.split("\0"))
        if len(path.parts) > 2 and path.name in PUBLIC_FILES and (ROOT / path).is_file()
    }
    run_folders = {path.parent for path in candidates if path.name in DATA_FILES}
    files = []
    for path in sorted(candidates):
        if path.parent not in run_folders:
            continue
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
        files.append(path.as_posix())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    if SITE.is_relative_to(output):
        parser.error("--output must be a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("app.js", "style.css"):
        shutil.copyfile(SITE / filename, output / filename)
    payload = (json.dumps({"files": fallback_files(output)}, ensure_ascii=False) + "\n").encode("utf-8")
    script = (SITE / "app.js").read_bytes()
    revision = hashlib.sha256(payload + script).hexdigest()[:12]
    data_file = f"data-{revision}.json"
    script_file = f"app-{revision}.js"
    (output / "data.json").write_bytes(payload)
    (output / data_file).write_bytes(payload)
    (output / script_file).write_bytes(script)
    html = (SITE / "index.html").read_text(encoding="utf-8").replace('src="app.js"', f'src="{script_file}"', 1)
    html = html.replace('data-results="data.json"', f'data-results="{data_file}"', 1)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"GitHub Pages dashboard built at {output}")


if __name__ == "__main__":
    main()
