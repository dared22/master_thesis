"""Run experiment notebooks with this interpreter in fresh, separate kernels."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def run_notebook(path: Path) -> None:
    """Execute in order, checkpoint outputs, and propagate any cell failure."""
    path = path.resolve()
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    notebook.metadata.pop("experiment_execution", None)
    nbformat.write(notebook, path)
    manager = KernelManager(kernel_name="python3")
    # Do not silently use a registered kernel from another project.
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]

    def on_start(cell: dict, cell_index: int) -> None:
        if cell["cell_type"] == "code" and cell["source"].strip():
            print(f"{datetime.now().isoformat(timespec='seconds')} "
                  f"{path.name}: executing cell {cell_index}", flush=True)

    def on_executed(cell: dict, cell_index: int, execute_reply: dict) -> None:
        nbformat.write(notebook, path)
        for output in cell.get("outputs", []):
            if output["output_type"] == "stream":
                print(output["text"], end="", flush=True)

    client = NotebookClient(
        notebook, km=manager, timeout=7200, allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_start=on_start, on_cell_executed=on_executed,
    )
    try:
        client.execute()
        notebook.metadata["experiment_execution"] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "interpreter": sys.executable,
            "all_cells_completed": True,
        }
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"Completed {path.name}", flush=True)


def main() -> None:
    """Run specified notebooks, or both experiment notebooks by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    args = parser.parse_args()
    sandbox = Path(__file__).resolve().parent
    paths = args.notebooks or [
        sandbox / "likelihood_calc.ipynb", sandbox / "sde_test_data.ipynb"
    ]
    for path in paths:
        run_notebook(path)


if __name__ == "__main__":
    main()
