"""Run EDA notebooks in order with the project Python interpreter."""

import sys
from pathlib import Path

import nbformat
from nbformat import NotebookNode
from jupyter_client import KernelManager
from nbclient import NotebookClient


def print_cell_output(cell: NotebookNode, **_: object) -> None:
    """Show notebook progress and export messages in the terminal."""
    for output in cell.get("outputs", []):
        if output.output_type == "stream" and output.name == "stdout":
            print(output.text, end="", flush=True)


def main() -> None:
    """Run the numbered notebooks, or the paths supplied on the command line."""
    eda = Path(__file__).resolve().parent
    paths = [Path(arg).resolve() for arg in sys.argv[1:]] or [
        eda / "01_prepare_price_histories.ipynb",
        eda / "02_sector_dependence.ipynb",
    ]
    for path in paths:
        print(f"Running {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        client = NotebookClient(
            notebook, km=manager, timeout=3600, on_cell_executed=print_cell_output
        )
        client.execute(cwd=str(path.parent), cleanup_kc=True)
        print(f"Finished {path.name}", flush=True)


if __name__ == "__main__":
    main()
