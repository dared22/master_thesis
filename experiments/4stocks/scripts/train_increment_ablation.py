"""Train one increment ablation for a specified seed."""

import argparse
from dataclasses import asdict, replace
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


def main() -> None:
    """Fit the unchanged baseline configuration with exactly one loss change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('gp5', 'absolute_acf_1'))
    parser.add_argument('seed', type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    results = DIRECTORY / 'results/sde_gan'
    baseline = results / 'increments' / f'seed_{args.seed}'
    config = SDEGANConfig(**json.loads((baseline / 'config.json').read_text()))
    assert config.generator_output == 'increments' and config.generator_steps == 2000
    assert config.gradient_penalty_weight == 10 and config.temporal_penalty_weight == 0
    assert config.signature_loss_weight == config.absolute_temporal_penalty_weight == 0
    config = (replace(config, gradient_penalty_weight=5)
              if args.experiment == 'gp5' else
              replace(config, absolute_temporal_penalty_weight=1))
    folder = 'increments_gp5' if args.experiment == 'gp5' else 'increments_absolute_acf_1'
    run_dir = results / folder / f'seed_{args.seed}'
    preprocessing = json.loads((DIRECTORY / 'data/preprocessing.json').read_text())
    assert preprocessing == json.loads((baseline / 'preprocessing.json').read_text())
    with np.load(DIRECTORY / 'data/prepared_windows.npz', allow_pickle=False) as prepared:
        training = prepared['train_windows']
        validation = prepared['validation_windows']
    protocol = {
        'experiment': args.experiment, 'seed': args.seed, 'config': asdict(config),
        'selection': 'lowest_validation_mmd2', 'sample_seed': 100, 'evaluation_paths': 1000,
        'train_sha256': hashlib.sha256(training.tobytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(validation.tobytes()).hexdigest(),
        'preprocessing': preprocessing,
        'code_sha256': hashlib.sha256(b''.join(
            (MODEL_DIR / name).read_bytes() for name in ('training.py', 'architecture.py', 'sde_gan.py')
        )).hexdigest(),
    }
    completion = run_dir / 'completion.json'
    if completion.is_file():
        previous = json.loads(completion.read_text())
        assert all(previous[key] == value for key, value in protocol.items())
        logging.info('Already completed: %s', run_dir)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'experiment.json').write_text(json.dumps(protocol, indent=2) + '\n')
    offset = np.asarray(preprocessing['training_mean']) / np.asarray(preprocessing['training_std'])
    logging.info('START %s seed=%d; GP=%g absolute_ACF=%g', args.experiment, args.seed,
                 config.gradient_penalty_weight, config.absolute_temporal_penalty_weight)
    started = perf_counter()
    model = SDEGAN(config)
    model.fit(training, validation, seed=args.seed, absolute_return_offset=offset)
    model.save(run_dir)
    for path in [run_dir, *sorted((run_dir / 'checkpoints').glob('step_*'))]:
        (path / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2) + '\n')
    expected = model.sample(17, 100)
    np.testing.assert_array_equal(expected, SDEGAN.load(run_dir).sample(17, 100))
    elapsed = perf_counter() - started
    completion.write_text(json.dumps({**protocol, 'elapsed_seconds': elapsed,
                                      'selected_step': model.run_metadata['selected_step'],
                                      'validation_mmd2': model.run_metadata['validation_score']},
                                     indent=2, allow_nan=False) + '\n')
    logging.info('DONE %s seed=%d selected_step=%d mmd2=%.6f elapsed=%.1fs',
                 args.experiment, args.seed, model.run_metadata['selected_step'],
                 model.run_metadata['validation_score'], elapsed)


if __name__ == '__main__':
    main()
