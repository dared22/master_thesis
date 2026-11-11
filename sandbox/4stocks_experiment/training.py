"""WGAN-GP updates, fixed MMD validation, and in-memory checkpoint selection."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from architecture import Critic, Generator

History = list[dict[str, int | float | None]]


@dataclass(frozen=True)
class SDEGANConfig:
    """Pilot settings; the observable contract remains 20 sessions by four stocks."""

    generator_hidden_size: int = 8
    initial_noise_size: int = 4
    brownian_size: int = 4
    critic_hidden_size: int = 16
    mlp_width: int = 32
    mlp_depth: int = 2
    generator_dt: float = 0.25
    critic_dt: float = 0.25
    brownian_tol: float = 0.0625
    batch_size: int = 8
    generator_lr: float = 1e-4
    critic_lr: float = 1e-4
    adam_b1: float = 0.0
    adam_b2: float = 0.9
    gradient_penalty_weight: float = 10.0
    temporal_penalty_weight: float = 0.0
    retain_validation_checkpoints: bool = False
    generator_steps: int = 2000
    critic_steps: int = 5
    validation_every: int = 100
    validation_size: int = 256
    evaluation_paths: int = 256
    bandwidth_sample_size: int = 256
    evaluation_seed: int = 100
    sampling_batch_size: int = 8

    def __post_init__(self) -> None:
        for name in (
            'generator_hidden_size', 'initial_noise_size', 'brownian_size',
            'critic_hidden_size', 'mlp_width', 'mlp_depth', 'batch_size',
            'generator_steps', 'critic_steps', 'validation_every',
            'validation_size', 'evaluation_paths', 'bandwidth_sample_size',
            'sampling_batch_size',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('generator_dt', 'critic_dt'):
            value = getattr(self, name)
            if (not np.isfinite(value) or value <= 0 or value > 1
                    or not np.isclose(1 / value, round(1 / value), rtol=0, atol=1e-7)):
                raise ValueError(f'{name} must divide one session exactly')
        if not 0 < self.brownian_tol < self.generator_dt:
            raise ValueError('brownian_tol must be positive and smaller than generator_dt')
        for name in ('generator_lr', 'critic_lr', 'gradient_penalty_weight'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(self.temporal_penalty_weight) or self.temporal_penalty_weight < 0:
            raise ValueError('temporal_penalty_weight must be nonnegative and finite')
        if not isinstance(self.retain_validation_checkpoints, bool):
            raise ValueError('retain_validation_checkpoints must be a boolean')
        if not (0 <= self.adam_b1 < 1 and 0 <= self.adam_b2 < 1):
            raise ValueError('Adam moments must lie in [0, 1)')
        if isinstance(self.evaluation_seed, bool) or not isinstance(self.evaluation_seed, int):
            raise ValueError('evaluation_seed must be an integer')


def validate_windows(values: Any, name: str) -> jax.Array:
    """Validate prepared floating windows, including finiteness after conversion."""
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (20, 4) or not len(values):
        raise ValueError(f'{name} must have nonempty shape (n_windows, 20, 4)')
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f'{name} must contain real floating-point values')
    with np.errstate(over='ignore'):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must contain finite float32 values')
    return jnp.asarray(values)


def initialize_models(config: SDEGANConfig, seed: int) -> tuple[Generator, Critic]:
    """Initialize players from streams distinct from the training stream."""
    key = jr.PRNGKey(seed)
    generator = Generator(
        config.generator_hidden_size, config.initial_noise_size,
        config.brownian_size, config.mlp_width, config.mlp_depth,
        key=jr.fold_in(key, 0),
    )
    critic = Critic(config.critic_hidden_size, config.mlp_width, config.mlp_depth,
                    key=jr.fold_in(key, 1))
    return generator, critic


@eqx.filter_jit
def generate_batch(
    generator: Generator, keys: jax.Array, dt: float, brownian_tol: float,
) -> jax.Array:
    """Vectorize independent forward solves over explicit path keys."""
    return jax.vmap(lambda key: generator(key, dt, brownian_tol))(keys)


def sample_paths(
    generator: Generator, n_paths: int, key: jax.Array, config: SDEGANConfig,
) -> np.ndarray:
    """Collect bounded batches on the host; keys depend on logical path indices."""
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError('n_paths must be a positive integer')
    batches = []
    size = config.sampling_batch_size
    for start in range(0, n_paths, size):
        indices = jnp.arange(start, start + size, dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jr.fold_in(key, index))(indices)
        paths = np.asarray(generate_batch(generator, keys, config.generator_dt,
                                           config.brownian_tol))
        batches.append(paths[:min(size, n_paths - start)])
    return np.concatenate(batches)


def gradient_penalty(
    critic: Critic, real: jax.Array, fake: jax.Array, alpha: jax.Array, dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Penalize the complete discretized critic's per-window input gradient."""
    interpolated = alpha[:, None, None] * real + (1 - alpha[:, None, None]) * jax.lax.stop_gradient(fake)
    gradients = jax.vmap(jax.grad(lambda x: critic(x, dt)))(interpolated)
    norms = jnp.linalg.norm(gradients.reshape(len(real), -1), axis=1)
    return jnp.mean((norms - 1) ** 2), jnp.mean(norms)


def critic_loss(
    critic: Critic, real: jax.Array, fake: jax.Array, alpha: jax.Array,
    config: SDEGANConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Wasserstein critic objective, with detached generated observations."""
    fake = jax.lax.stop_gradient(fake)
    penalty, input_norm = gradient_penalty(critic, real, fake, alpha, config.critic_dt)
    scores = jax.vmap(lambda x: critic(x, config.critic_dt))
    loss = scores(fake).mean() - scores(real).mean() + config.gradient_penalty_weight * penalty
    return loss, (penalty, input_norm)


def temporal_autocorrelation(values: jax.Array) -> jax.Array:
    """Pooled within-path Pearson correlations at lags 1–5, per stock.

    Center over all valid pairs at each lag, never across path boundaries.
    A variance floor in standardized units keeps constant batches and their
    gradients finite; zero-variance channels have a stabilized value of zero.
    """
    correlations = []
    for lag in range(1, 6):
        left, right = values[:, :-lag], values[:, lag:]
        left = left - left.mean(axis=(0, 1))
        right = right - right.mean(axis=(0, 1))
        denominator = jnp.sqrt(
            jnp.maximum(jnp.mean(left ** 2, axis=(0, 1)), 1e-6)
            * jnp.maximum(jnp.mean(right ** 2, axis=(0, 1)), 1e-6))
        correlations.append(jnp.mean(left * right, axis=(0, 1)) / denominator)
    return jnp.stack(correlations)


def temporal_penalty(paths: jax.Array, target: jax.Array) -> jax.Array:
    """Match a fixed training-only target, averaged over five lags and stocks."""
    return jnp.mean((temporal_autocorrelation(paths) - jax.lax.stop_gradient(target)) ** 2)


def _generator_objective(
    generator: Generator, critic: Critic, keys: jax.Array, config: SDEGANConfig,
    temporal_target: jax.Array | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Compute both generator terms from the same generated batch."""
    paths = generate_batch(generator, keys, config.generator_dt, config.brownian_tol)
    adversarial = -jax.vmap(lambda x: critic(x, config.critic_dt))(paths).mean()
    penalty = jnp.array(0.0)
    if config.temporal_penalty_weight > 0:
        if temporal_target is None:
            raise ValueError('A training temporal_target is required when the penalty is enabled')
        penalty = temporal_penalty(paths, temporal_target)
    return adversarial + config.temporal_penalty_weight * penalty, (adversarial, penalty)


def generator_loss(
    generator: Generator, critic: Critic, keys: jax.Array, config: SDEGANConfig,
    temporal_target: jax.Array | None = None,
) -> jax.Array:
    """Differentiate the fixed critic and optional temporal penalty through paths."""
    return _generator_objective(generator, critic, keys, config, temporal_target)[0]


def tree_finite(tree: Any) -> jax.Array:
    """Check all numerical leaves without transferring parameter arrays to the host."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))


def make_updates(
    config: SDEGANConfig, temporal_target: jax.Array | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build independent Adam transformations and compiled player updates."""
    g_optimizer = optax.adam(config.generator_lr, b1=config.adam_b1, b2=config.adam_b2)
    d_optimizer = optax.adam(config.critic_lr, b1=config.adam_b1, b2=config.adam_b2)

    @eqx.filter_jit
    def critic_update(critic: Critic, state: Any, real: jax.Array,
                      fake: jax.Array, alpha: jax.Array) -> tuple[Any, ...]:
        (loss, (penalty, input_norm)), grads = eqx.filter_value_and_grad(
            critic_loss, has_aux=True
        )(critic, real, fake, alpha, config)
        updates, state = d_optimizer.update(grads, state)
        critic = eqx.apply_updates(critic, updates)
        metrics = jnp.array([loss, penalty, optax.tree.norm(grads), input_norm])
        valid = tree_finite((metrics, grads, critic, state))
        return critic, state, metrics, valid

    @eqx.filter_jit
    def generator_update(generator: Generator, state: Any, critic: Critic,
                         keys: jax.Array) -> tuple[Any, ...]:
        (loss, (adversarial, penalty)), grads = eqx.filter_value_and_grad(
            _generator_objective, has_aux=True
        )(generator, critic, keys, config, temporal_target)
        updates, state = g_optimizer.update(grads, state)
        generator = eqx.apply_updates(generator, updates)
        metrics = jnp.array([loss, optax.tree.norm(grads), adversarial, penalty])
        valid = tree_finite((metrics, grads, generator, state))
        return generator, state, metrics, valid

    return g_optimizer, d_optimizer, generator_update, critic_update


def kernel_bandwidth(train: np.ndarray, key: jax.Array, size: int) -> float:
    """Median positive pair distance in a seeded training-only sample."""
    indices = np.asarray(jr.choice(key, len(train), (min(size, len(train)),), replace=False))
    flat = np.asarray(train)[indices].reshape(len(indices), 80).astype(np.float64)
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    distances = distances[np.triu_indices(len(flat), k=1)]
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('MMD bandwidth requires at least one positive training pair distance')
    return float(np.median(positive))


def mmd_squared(real: np.ndarray, fake: np.ndarray, bandwidth: float) -> float:
    """Biased squared Gaussian-kernel MMD, including all diagonal terms."""
    real = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
    fake = np.asarray(fake, dtype=np.float64).reshape(len(fake), -1)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        squared = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
        return float(np.exp(-squared / (2 * bandwidth ** 2)).mean())

    return kernel_mean(real, real) + kernel_mean(fake, fake) - 2 * kernel_mean(real, fake)


def volatility_error(real: np.ndarray, fake: np.ndarray) -> float:
    """RMS relative error of pooled per-stock sample volatility, in log units.

    Standardization cancels in this ratio. A 1e-6 denominator floor handles
    constant validation channels without a division by zero.
    """
    real_std = np.asarray(real, dtype=np.float64).reshape(-1, 4).std(axis=0, ddof=1)
    fake_std = np.asarray(fake, dtype=np.float64).reshape(-1, 4).std(axis=0, ddof=1)
    return float(np.sqrt(np.mean(((fake_std - real_std) / np.maximum(real_std, 1e-6)) ** 2)))


def train(
    generator: Generator, critic: Critic, train_windows: Any,
    validation_windows: Any, config: SDEGANConfig, seed: int,
) -> dict[str, Any]:
    """Fit fresh optimizer states, retaining the strict best fixed-validation model.

    No data or checkpoints are written. History has one row per generator cycle;
    cycle_seconds excludes validation, and the first cycle includes compilation.
    """
    started = perf_counter()
    data = validate_windows(train_windows, 'train_windows')
    validation = validate_windows(validation_windows, 'validation_windows')
    bandwidth_key, subset_key, evaluation_key = jr.split(jr.PRNGKey(config.evaluation_seed), 3)
    bandwidth = kernel_bandwidth(np.asarray(data), bandwidth_key, config.bandwidth_sample_size)
    indices = jr.choice(subset_key, len(validation),
                        (min(config.validation_size, len(validation)),), replace=False)
    fixed_validation = np.asarray(validation[indices])
    temporal_target = (jax.lax.stop_gradient(temporal_autocorrelation(data))
                       if config.temporal_penalty_weight > 0 else None)
    validation_target = temporal_autocorrelation(jnp.asarray(fixed_validation))
    g_opt, d_opt, g_update, d_update = make_updates(config, temporal_target)
    g_state = g_opt.init(eqx.filter(generator, eqx.is_inexact_array))
    d_state = d_opt.init(eqx.filter(critic, eqx.is_inexact_array))
    key = jr.fold_in(jr.PRNGKey(seed), 2)

    def evaluate(model: Generator) -> tuple[float, float, float]:
        samples = sample_paths(model, config.evaluation_paths, evaluation_key, config)
        score = mmd_squared(fixed_validation, samples, bandwidth)
        temporal_error = float(temporal_penalty(jnp.asarray(samples), validation_target))
        vol_error = volatility_error(fixed_validation, samples)
        if not np.isfinite([score, temporal_error, vol_error]).all():
            raise FloatingPointError('Nonfinite validation MMD2, temporal error, or volatility error')
        return score, temporal_error, vol_error

    best_score, temporal_error, vol_error = evaluate(generator)
    checkpoints = []
    if config.retain_validation_checkpoints:
        checkpoints.append({'step': 0, 'generator': generator, 'validation_mmd2': best_score,
                            'validation_temporal_error': temporal_error,
                            'validation_volatility_error': vol_error})
    best_generator, best_step = generator, 0
    history: History = [{
        'step': 0, 'generator_loss': None, 'critic_loss': None,
        'gradient_penalty': None, 'generator_grad_norm': None,
        'critic_grad_norm': None, 'input_grad_norm': None,
        'validation_mmd2': best_score, 'elapsed_seconds': perf_counter() - started,
        'cycle_seconds': None,
        'generator_adversarial_loss': None, 'temporal_penalty': None,
        'validation_temporal_error': temporal_error,
        'validation_volatility_error': vol_error,
    }]
    for step in range(1, config.generator_steps + 1):
        cycle_started = perf_counter()
        critic_metrics = []
        for substep in range(1, config.critic_steps + 1):
            key, batch_key, fake_key, alpha_key = jr.split(key, 4)
            real = data[jr.randint(batch_key, (config.batch_size,), 0, len(data))]
            try:
                fake = generate_batch(generator, jr.split(fake_key, config.batch_size),
                                      config.generator_dt, config.brownian_tol)
                critic, d_state, metrics, valid = d_update(
                    critic, d_state, real, fake, jr.uniform(alpha_key, (config.batch_size,)))
                if not bool(valid):
                    raise FloatingPointError('Nonfinite critic update')
            except Exception as error:
                raise RuntimeError(f'Generator step {step}, critic substep {substep} failed') from error
            critic_metrics.append(np.asarray(metrics))
        key, fake_key = jr.split(key)
        try:
            generator, g_state, metrics, valid = g_update(
                generator, g_state, critic, jr.split(fake_key, config.batch_size))
            if not bool(valid):
                raise FloatingPointError('Nonfinite generator update')
        except Exception as error:
            raise RuntimeError(f'Generator update {step} failed') from error
        g_loss, g_norm, adversarial, g_temporal_penalty = map(float, metrics)
        cycle_seconds = perf_counter() - cycle_started
        d_loss, penalty, d_norm, input_norm = map(float, np.mean(critic_metrics, axis=0))
        score = None
        temporal_error = None
        vol_error = None
        if step % config.validation_every == 0 or step == config.generator_steps:
            score, temporal_error, vol_error = evaluate(generator)
            if config.retain_validation_checkpoints:
                checkpoints.append({'step': step, 'generator': generator, 'validation_mmd2': score,
                                    'validation_temporal_error': temporal_error,
                                    'validation_volatility_error': vol_error})
            if score < best_score:
                best_generator, best_score, best_step = generator, score, step
        history.append({
            'step': step, 'generator_loss': g_loss, 'critic_loss': d_loss,
            'gradient_penalty': penalty, 'generator_grad_norm': g_norm,
            'critic_grad_norm': d_norm, 'input_grad_norm': input_norm,
            'validation_mmd2': score, 'elapsed_seconds': perf_counter() - started,
            'cycle_seconds': cycle_seconds,
            'generator_adversarial_loss': adversarial,
            'temporal_penalty': g_temporal_penalty,
            'validation_temporal_error': temporal_error,
            'validation_volatility_error': vol_error,
        })
    return {
        'best_generator': best_generator, 'best_step': best_step,
        'validation_score': best_score, 'kernel_bandwidth': bandwidth,
        'final_generator': generator, 'final_critic': critic, 'history': history,
        'temporal_target': None if temporal_target is None else np.asarray(temporal_target).tolist(),
        'validation_checkpoints': checkpoints,
    }
