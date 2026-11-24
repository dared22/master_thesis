"""Train the stabilized, mean-anchored increments SDE-GAN baseline on 4stocks data."""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_DIR = DIRECTORY / 'model'
sys.path.insert(0, str(MODEL_DIR))
from sde_gan import SDEGAN, SDEGANConfig


def build_baseline_config(
    validation_size: int,
    *,
    generator_steps: int = 2000,
    mean_penalty_weight: float = 10.0,
) -> SDEGANConfig:
    """Return the stabilized, mean-anchored increments baseline configuration."""
    return SDEGANConfig(
        generator_output='increments',
        batch_size=64,
        sampling_batch_size=64,
        grad_clip_norm=10.0,
        lr_schedule='cosine',
        ema_decay=0.99,
        mean_penalty_weight=mean_penalty_weight,
        validation_size=validation_size,
        evaluation_paths=1000,
        mmd_bandwidth_multipliers=(0.5, 1.0, 2.0),
        generator_steps=generator_steps,
        validation_every=100,
        retain_validation_checkpoints=True,
    )


def train_seed(
    seed: int,
    *,
    generator_steps: int = 2000,
    mean_penalty_weight: float = 10.0,
    force: bool = False,
) -> Path:
    """Fit one seed and save selected + validation-snapshot checkpoints."""
    preprocessing = json.loads((DIRECTORY / 'data/preprocessing.json').read_text())
    with np.load(DIRECTORY / 'data/prepared_windows.npz', allow_pickle=False) as prepared:
        training = prepared['train_windows']
        validation = prepared['validation_windows']

    config = build_baseline_config(
        len(validation),
        generator_steps=generator_steps,
        mean_penalty_weight=mean_penalty_weight,
    )
    run_dir = DIRECTORY / 'results/sde_gan' / f'seed_{seed}'
    protocol = {
        'seed': seed,
        'config': asdict(config),
        'selection': 'lowest_validation_mmd2',
        'train_sha256': hashlib.sha256(training.tobytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(validation.tobytes()).hexdigest(),
        'preprocessing': preprocessing,
        'code_sha256': hashlib.sha256(b''.join(
            (MODEL_DIR / name).read_bytes() for name in ('training.py', 'architecture.py', 'sde_gan.py')
        )).hexdigest(),
    }
    completion = run_dir / 'completion.json'
    if completion.is_file() and not force:
        previous = json.loads(completion.read_text())
        if all(previous.get(key) == value for key, value in protocol.items()):
            logging.info('Already completed matching run: %s', run_dir)
            return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'experiment.json').write_text(json.dumps(protocol, indent=2) + '\n')
    offset = np.asarray(preprocessing['training_mean']) / np.asarray(preprocessing['training_std'])
    logging.info('START seed=%d steps=%d batch=%d mean_weight=%g',
                 seed, config.generator_steps, config.batch_size, config.mean_penalty_weight)
    started = perf_counter()
    model = SDEGAN(config)
    model.fit(training, validation, seed=seed, absolute_return_offset=offset)
    model.save(run_dir)
    for path in [run_dir, *sorted((run_dir / 'checkpoints').glob('step_*'))]:
        (path / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2) + '\n')
    np.testing.assert_array_equal(model.sample(17, 100), SDEGAN.load(run_dir).sample(17, 100))
    elapsed = perf_counter() - started
    completion.write_text(json.dumps({
        **protocol,
        'elapsed_seconds': elapsed,
        'selected_step': model.run_metadata['selected_step'],
        'validation_mmd2': model.run_metadata['validation_score'],
    }, indent=2, allow_nan=False) + '\n')
    logging.info('DONE seed=%d selected_step=%d mmd2=%.6f elapsed=%.1fs',
                 seed, model.run_metadata['selected_step'],
                 model.run_metadata['validation_score'], elapsed)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seeds', type=int, nargs='+', help='Training seed(s), e.g. 0 1 2.')
    parser.add_argument('--steps', type=int, default=2000, help='Generator update steps.')
    parser.add_argument('--mean-weight', type=float, default=10.0, help='Unbiased mean-anchoring penalty weight.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing matching run.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for seed in args.seeds:
        train_seed(seed, generator_steps=args.steps, mean_penalty_weight=args.mean_weight, force=args.force)


if __name__ == '__main__':
    main()
