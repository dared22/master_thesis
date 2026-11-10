"""Notebook-facing SDE-GAN API for prepared standardized log-return windows."""

import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

import equinox as eqx
import jax.random as jr
import numpy as np

from architecture import Generator
from training import (
    History, SDEGANConfig, initialize_models, sample_paths, train,
)

__all__ = ['SDEGAN', 'SDEGANConfig']


class SDEGAN:
    """Own a selected Equinox generator; refitting always starts a fresh run.

    Inputs and samples have shape (n, 20, 4), in standardized log-return units.
    History is a list of scalar records suitable for pandas.DataFrame(history).
    Checkpoints restore inference only, not optimizer states.
    """

    def __init__(self, config: SDEGANConfig | None = None) -> None:
        self.config = SDEGANConfig() if config is None else config
        self.generator: Generator | None = None
        self.history: History = []
        self.run_metadata: dict[str, Any] = {}
        self.training_result: dict[str, Any] | None = None

    def fit(self, train_windows: Any, validation_windows: Any,
            seed: int = 0) -> History:
        """Fit from scratch and retain the best fixed-MMD checkpoint in memory."""
        generator, critic = initialize_models(self.config, seed)
        result = train(generator, critic, train_windows, validation_windows, self.config, seed)
        self.generator = result['best_generator']
        self.training_result = result
        self.history = result['history']
        self.run_metadata = {
            'fit_seed': seed, 'evaluation_seed': self.config.evaluation_seed,
            'selected_step': result['best_step'],
            'validation_score': result['validation_score'],
            'kernel_bandwidth': result['kernel_bandwidth'],
            'package_versions': {name: version(name) for name in
                                 ('jax', 'jaxlib', 'equinox', 'diffrax', 'optax', 'numpy')},
            'python_version': platform.python_version(),
        }
        return self.history

    def _require_generator(self) -> None:
        if self.generator is None:
            raise RuntimeError('Fit or load an SDEGAN before sampling or saving')

    def sample(self, n_paths: int, seed: int) -> np.ndarray:
        """Generate paths using an explicit seed, without changing random state."""
        self._require_generator()
        return sample_paths(self.generator, n_paths, jr.PRNGKey(seed), self.config)

    def save(self, directory: str | Path) -> None:
        """Write the selected generator, configuration, run metadata, and history."""
        self._require_generator()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(directory / 'generator.eqx', self.generator)
        for filename, contents in (('config.json', asdict(self.config)),
                                   ('run.json', self.run_metadata)):
            (directory / filename).write_text(json.dumps(contents, indent=2, allow_nan=False) + '\n')
        with (directory / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)

    @classmethod
    def load(cls, directory: str | Path) -> 'SDEGAN':
        """Restore an inference checkpoint in the same software environment."""
        directory = Path(directory)
        model = cls(SDEGANConfig(**json.loads((directory / 'config.json').read_text())))
        template, _ = initialize_models(model.config, 0)
        model.generator = eqx.tree_deserialise_leaves(directory / 'generator.eqx', template)
        model.run_metadata = json.loads((directory / 'run.json').read_text())
        with (directory / 'history.csv').open(newline='') as handle:
            model.history = [
                {key: None if value == '' else int(value) if key == 'step' else float(value)
                 for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        return model
