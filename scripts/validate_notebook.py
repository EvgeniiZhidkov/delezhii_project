"""Execute the notebook using this interpreter, without installing a user kernel."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("chores_e1_3.ipynb"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    with TemporaryDirectory(prefix="e1-notebook-check-") as temporary:
        runtime = Path(temporary)
        kernel_dir = runtime / "jupyter" / "kernels" / "e1-validation"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "E1 validation", "language": "python",
        }), encoding="utf-8")
        os.environ.update(JUPYTER_PATH=str(runtime / "jupyter"),
                          JUPYTER_RUNTIME_DIR=str(runtime / "runtime"),
                          IPYTHONDIR=str(runtime / "ipython"), MPLCONFIGDIR=str(runtime / "matplotlib"))
        import nbformat
        from nbclient import NotebookClient
        notebook = nbformat.read(root / "chores_e1_3.ipynb", as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=180, kernel_name="e1-validation",
                       resources={"metadata": {"path": str(root)}}).execute()
        notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbformat.write(notebook, args.output)
        images = sum("image/png" in output.get("data", {}) for cell in notebook.cells
                     for output in cell.get("outputs", []))
        print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells; {images} figures; saved {args.output}")


if __name__ == "__main__":
    main()
