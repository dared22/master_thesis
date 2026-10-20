"""Export notebook result tables and the environment needed to reproduce them."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def json_scalar(value: Any) -> Any:
    """Convert NumPy scalar table entries without changing their values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def source_digest(path: Path) -> str:
    """Hash notebook sources independently of changing execution outputs."""
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        content = json.dumps(
            [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]],
            ensure_ascii=False,
        ).encode()
    else:
        content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def export_results(
    notebook_name: str,
    tables: dict[str, list[dict[str, Any]]],
    *,
    seeds: dict[str, Any],
    settings: dict[str, Any],
) -> Path:
    """Write CSV tables, full-precision JSON, and execution provenance."""
    sandbox = Path(__file__).resolve().parent
    output = sandbox / "experiment_results" / Path(notebook_name).stem
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not rows:
            raise ValueError(f"Cannot export an empty result table: {name}")
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    sources = [sandbox / notebook_name, Path(__file__).resolve()]
    if notebook_name == "sde_test_data.ipynb":
        sources.append(sandbox / "train_sde.py")
    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
        )),
        "seeds": seeds,
        "settings": settings,
        "source_sha256": {path.name: source_digest(path) for path in sources},
    }
    (output / "results.json").write_text(
        json.dumps({"metadata": metadata, "tables": tables}, indent=2,
                   allow_nan=False, default=json_scalar) + "\n"
    )
    print(f"Exported result tables and execution environment to {output}")
    return output
