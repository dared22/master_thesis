"""Run uv run python -m unittest from this directory.

TestSmokeRun performs the full 50-cycle CPU acceptance run. The remaining
classes provide faster numerical, gradient, and API regression checks.
"""

from dataclasses import replace
import csv
import json
from pathlib import Path
import resource
import tempfile
import unittest
from unittest.mock import patch

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from architecture import solve_sde
from sde_gan import SDEGAN, SDEGANConfig
from training import (
    generate_batch, generator_loss, gradient_penalty, initialize_models,
    kernel_bandwidth, make_updates, mmd_squared, sample_paths, train,
    tree_finite, validate_windows,
    temporal_autocorrelation, temporal_penalty, absolute_temporal_penalty,
    volatility_error, signature_features, fit_signature_reference, signature_loss, signature_error,
)


def small_config(**kwargs: object) -> SDEGANConfig:
    return replace(SDEGANConfig(
        generator_hidden_size=2, critic_hidden_size=2, brownian_size=2,
        mlp_width=4, mlp_depth=1, batch_size=2, sampling_batch_size=2,
        generator_steps=1, critic_steps=1, evaluation_paths=4,
        validation_size=4,
    ), **kwargs)


def synthetic(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 20, 4))
    loading = np.array([[1, 0, 0, 0], [.5, .8, 0, 0],
                        [-.3, .2, .9, 0], [.4, -.2, .1, .8]])
    return (factors @ loading.T).astype(np.float32)


def assert_tree_equal(test: unittest.TestCase, a: object, b: object) -> None:
    leaves_a, structure_a = jax.tree_util.tree_flatten(a)
    leaves_b, structure_b = jax.tree_util.tree_flatten(b)
    test.assertEqual(structure_a, structure_b)
    for x, y in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(x, y)


class TestInputsAndMetrics(unittest.TestCase):
    def test_contract_and_config(self) -> None:
        for values in (np.zeros((0, 20, 4)), np.zeros((2, 19, 4)),
                       np.zeros((2, 20, 5)), np.full((2, 20, 4), np.nan),
                       np.full((2, 20, 4), np.inf), np.zeros((2, 20, 4), complex),
                       np.full((2, 20, 4), 1e300)):
            with self.assertRaises(ValueError):
                validate_windows(values, 'data')
        self.assertEqual(validate_windows(synthetic(0, 2).astype(float), 'data').dtype, jnp.float32)
        for kwargs in ({'batch_size': 0}, {'generator_dt': .3}, {'critic_dt': 0},
                       {'brownian_tol': .25}, {'adam_b1': 1}, {'generator_lr': np.nan}):
            with self.assertRaises(ValueError):
                small_config(**kwargs)
        model = SDEGAN()
        with self.assertRaises(RuntimeError):
            model.sample(2, seed=0)
        with self.assertRaises(RuntimeError):
            model.save('unused')

    def test_bandwidth_and_mmd(self) -> None:
        data = synthetic(3, 5)
        key = jr.PRNGKey(0)
        self.assertEqual(kernel_bandwidth(data, key, 4), kernel_bandwidth(data, key, 4))
        with self.assertRaisesRegex(ValueError, 'positive training pair'):
            kernel_bandwidth(np.zeros((5, 20, 4)), key, 5)
        # Exact duplicates do not contribute zero distances to the median.
        duplicate = np.zeros((3, 20, 4), dtype=np.float32)
        duplicate[-1] = 1
        self.assertAlmostEqual(kernel_bandwidth(duplicate, key, 3), np.sqrt(80))
        a, b = np.array([[0.], [1.]]), np.array([[2.], [3.], [4.]])
        kernel = lambda x, y: np.exp(-np.sum((x - y) ** 2) / 2)
        expected = (np.mean([kernel(x, y) for x in a for y in a])
                    + np.mean([kernel(x, y) for x in b for y in b])
                    - 2 * np.mean([kernel(x, y) for x in a for y in b]))
        self.assertAlmostEqual(mmd_squared(a, b, 1), expected)
        self.assertAlmostEqual(mmd_squared(data, data, 2), 0)

    def test_selection_ties_and_final_validation(self) -> None:
        config = small_config(generator_steps=3, validation_every=2)
        generator, critic = initialize_models(config, 0)
        # Exercise the real loop and optimizer wiring without compiling solves.
        g_opt, d_opt, _, _ = make_updates(config)
        def d_update(model, state, real, fake, alpha):
            return model, state, jnp.ones(4), jnp.array(True)
        def g_update(model, state, critic, keys):
            return model, state, jnp.ones(6), jnp.array(True)
        with patch('training.make_updates', return_value=(g_opt, d_opt, g_update, d_update)), \
             patch('training.generate_batch', return_value=jnp.zeros((2, 20, 4))), \
             patch('training.sample_paths', return_value=np.zeros((4, 20, 4))), \
             patch('training.mmd_squared', side_effect=[3., 2., 2.]):
            result = train(generator, critic, synthetic(0, 8), synthetic(1, 8), config, 0)
        self.assertEqual(result['best_step'], 2)
        self.assertEqual([r['step'] for r in result['history'] if r['validation_mmd2'] is not None], [0, 2, 3])

    def test_evaluation_is_shared_across_fit_seeds(self) -> None:
        config = small_config()
        real_evaluations, noise_keys = [], []
        g_opt, d_opt, _, _ = make_updates(config)
        def d_update(model, state, real, fake, alpha):
            return model, state, jnp.ones(4), jnp.array(True)
        def g_update(model, state, critic, keys):
            return model, state, jnp.ones(6), jnp.array(True)
        def sample(model, n_paths, key, config):
            noise_keys.append(np.asarray(key))
            return np.zeros((n_paths, 20, 4))
        def mmd(real, fake, bandwidth):
            real_evaluations.append((real.copy(), bandwidth))
            return 1.0
        with patch('training.make_updates', return_value=(g_opt, d_opt, g_update, d_update)), \
             patch('training.generate_batch', return_value=jnp.zeros((2, 20, 4))), \
             patch('training.sample_paths', side_effect=sample), \
             patch('training.mmd_squared', side_effect=mmd):
            for seed in (0, 1):
                generator, critic = initialize_models(config, seed)
                result = train(generator, critic, synthetic(0, 8), synthetic(1, 8), config, seed)
                self.assertEqual(result['best_step'], 0)
        for key in noise_keys[1:]:
            np.testing.assert_array_equal(key, noise_keys[0])
        for real, bandwidth in real_evaluations[1:]:
            np.testing.assert_array_equal(real, real_evaluations[0][0])
            self.assertEqual(bandwidth, real_evaluations[0][1])

    def test_nonfinite_update_reports_location(self) -> None:
        config = small_config()
        generator, critic = initialize_models(config, 0)
        g_opt, d_opt, _, _ = make_updates(config)
        def failed_update(model, state, real, fake, alpha):
            return model, state, jnp.full(4, jnp.nan), jnp.array(False)
        with patch('training.make_updates', return_value=(g_opt, d_opt, None, failed_update)), \
             patch('training.generate_batch', return_value=jnp.zeros((2, 20, 4))), \
             patch('training.sample_paths', return_value=np.zeros((4, 20, 4))):
            with self.assertRaisesRegex(RuntimeError, 'step 1, critic substep 1') as caught:
                train(generator, critic, synthetic(0, 8), synthetic(1, 8), config, 0)
        self.assertIsInstance(caught.exception.__cause__, FloatingPointError)


class TestNumerics(unittest.TestCase):
    def test_additive_sde_and_brownian_coupling(self) -> None:
        mu = jnp.array([.1, -.05])
        sigma = jnp.array([[.4, .2], [-.1, .3]])
        times = jnp.arange(20, dtype=jnp.float32)
        def solve(key, dt):
            bm = dfx.VirtualBrownianTree(0., 19., .0625, (2,), key)
            paths = solve_sde(lambda t, y, args: mu,
                              lambda t, y, args: sigma, jnp.zeros(2), bm, dt)
            exact = times[:, None] * mu + jax.vmap(lambda t: sigma @ bm.evaluate(t))(times)
            return paths, exact
        coarse, exact = jax.jit(lambda k: solve(k, .25))(jr.PRNGKey(7))
        fine, exact_fine = jax.jit(lambda k: solve(k, .125))(jr.PRNGKey(7))
        np.testing.assert_array_equal(exact, exact_fine)
        np.testing.assert_allclose(coarse, exact, atol=5e-6)
        np.testing.assert_allclose(coarse, fine, atol=5e-6)
        # Batched Monte Carlo also checks the joint covariance, not just marginals.
        paths, _ = jax.jit(jax.vmap(lambda k: solve(k, .25)))(jr.split(jr.PRNGKey(9), 2048))
        endpoints = np.asarray(paths[:, -1])
        covariance = 19 * np.asarray(sigma @ sigma.T)
        mean_se = np.sqrt(np.diag(covariance) / len(endpoints))
        self.assertTrue(np.all(np.abs(endpoints.mean(0) - 19 * np.asarray(mu)) < 5 * mean_se))
        covariance_se = np.sqrt((covariance ** 2 + np.outer(np.diag(covariance), np.diag(covariance))) / (len(endpoints) - 1))
        self.assertTrue(np.all(np.abs(np.cov(endpoints.T) - covariance) < 5 * covariance_se))


class TestIncrementGenerator(unittest.TestCase):
    def test_additive_process_returns_daily_increments(self) -> None:
        config = small_config(generator_output='increments', generator_hidden_size=4,
                              brownian_size=4)
        generator, _ = initialize_models(config, 0)
        mu = jnp.array([.1, -.05, .02, 0.])
        sigma = jnp.array([[.4, 0, 0, 0], [.2, .3, 0, 0],
                           [0, 0, .2, 0], [0, 0, .1, .3]])

        def constant(input_size, output_size, bias):
            network = eqx.nn.Linear(input_size, output_size, key=jr.PRNGKey(1))
            return eqx.tree_at(lambda n: (n.weight, n.bias), network,
                               (jnp.zeros_like(network.weight), bias))

        generator = eqx.tree_at(
            lambda g: (g.initial, g.drift, g.diffusion, g.readout.weight), generator,
            (constant(4, 4, jnp.zeros(4)), constant(5, 4, mu),
             constant(5, 16, sigma.ravel()), jnp.eye(4)),
        )
        key = jr.PRNGKey(19)
        brownian_key = jr.split(key)[1]
        brownian = dfx.VirtualBrownianTree(0., 20., .0625, (4,), brownian_key)
        levels = jax.vmap(lambda t: sigma @ brownian.evaluate(t))(
            jnp.arange(21, dtype=jnp.float32))
        expected = mu + jnp.diff(levels, axis=0)
        actual = eqx.filter_jit(lambda g: g(key))(generator)
        self.assertEqual(actual.shape, (20, 4))
        self.assertIsNone(generator.readout.bias)
        np.testing.assert_allclose(actual, expected, atol=5e-6)
        np.testing.assert_allclose(generator(key, dt=.125), expected, atol=5e-6)

    def test_training_gradients_sampling_and_restore(self) -> None:
        config = small_config(generator_output='increments', retain_validation_checkpoints=True)
        generator, critic = initialize_models(config, 4)
        keys = jr.split(jr.PRNGKey(5), config.batch_size)
        grads = eqx.filter_jit(eqx.filter_grad(generator_loss))(generator, critic, keys, config)
        self.assertTrue(bool(tree_finite(grads)))
        self.assertGreater(float(optax.tree.norm(grads.diffusion)), 0)
        self.assertGreater(float(optax.tree.norm(grads.readout)), 0)
        model = SDEGAN(config)
        model.fit(synthetic(6, 8), synthetic(7, 8), seed=4)
        paths = model.sample(5, 8)
        self.assertEqual(paths.shape, (5, 20, 4))
        self.assertTrue(np.isfinite(paths).all())
        np.testing.assert_array_equal(paths, model.sample(5, 8))
        alternate = sample_paths(model.generator, 5, jr.PRNGKey(8),
                                 replace(config, sampling_batch_size=3))
        np.testing.assert_array_equal(paths, alternate)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = SDEGAN.load(directory)
            self.assertEqual(restored.config.generator_output, 'increments')
            np.testing.assert_array_equal(paths, restored.sample(5, 8))
            snapshot = SDEGAN.load(Path(directory) / 'checkpoints/step_0001')
            self.assertEqual(snapshot.config.generator_output, 'increments')
            self.assertTrue(np.isfinite(snapshot.sample(2, 8)).all())

    def test_legacy_config_defaults_to_direct_returns(self) -> None:
        model = SDEGAN(small_config())
        model.fit(synthetic(9, 8), synthetic(10, 8), seed=0)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            config_path = Path(directory) / 'config.json'
            saved_config = json.loads(config_path.read_text())
            del saved_config['generator_output']
            config_path.write_text(json.dumps(saved_config))
            restored = SDEGAN.load(directory)
            self.assertEqual(restored.config.generator_output, 'returns')
            np.testing.assert_array_equal(model.sample(2, 11), restored.sample(2, 11))
        with self.assertRaises(ValueError):
            small_config(generator_output='prices')


class TestGradientsAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.generator, cls.critic = initialize_models(cls.config, 7)
        cls.real = jnp.asarray(synthetic(3, 2) * .1)
        cls.fake = jnp.asarray(synthetic(4, 2) * .1)
        cls.alpha = jnp.array([.3, .7])

    def test_full_penalty_derivatives(self) -> None:
        def penalty(critic):
            return gradient_penalty(critic, self.real, self.fake, self.alpha, .25)[0]
        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(penalty))(self.critic)
        self.assertTrue(bool(tree_finite((value, grads))))
        self.assertGreater(float(jnp.linalg.norm(grads.initial.layers[0].weight)), 0)
        # Move only initial-network weights: detects detachment of h(0).
        weight = self.critic.initial.layers[0].weight
        direction = jnp.ones_like(weight) / jnp.sqrt(weight.size)
        def moved(epsilon):
            return eqx.tree_at(lambda c: c.initial.layers[0].weight,
                               self.critic, weight + epsilon * direction)
        expected = float(jnp.sum(grads.initial.layers[0].weight * direction))
        evaluate = eqx.filter_jit(penalty)
        actual = float((evaluate(moved(.001)) - evaluate(moved(-.001))) / .002)
        np.testing.assert_allclose(actual, expected, rtol=.03, atol=3e-4)
        # The input gradient uses all coordinates, per window (not a batch-mean score).
        mixed = self.alpha[:, None, None] * self.real + (1 - self.alpha[:, None, None]) * self.fake
        norms = jax.vmap(lambda x: jnp.linalg.norm(jax.grad(self.critic)(x)))(mixed)
        np.testing.assert_allclose(value, jnp.mean((norms - 1) ** 2), rtol=1e-5)

    def test_generator_gradients_and_update_isolation(self) -> None:
        keys = jr.split(jr.PRNGKey(3), 2)
        gradients = eqx.filter_jit(eqx.filter_grad(generator_loss))(
            self.generator, self.critic, keys, self.config)
        self.assertTrue(bool(tree_finite(gradients)))
        for network in (gradients.initial, gradients.drift, gradients.diffusion, gradients.readout):
            self.assertGreater(float(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(network))), 0)
        g_opt, d_opt, g_update, d_update = make_updates(self.config)
        gs = g_opt.init(eqx.filter(self.generator, eqx.is_inexact_array))
        ds = d_opt.init(eqx.filter(self.critic, eqx.is_inexact_array))
        saved_generator, saved_gs = self.generator, gs
        critic, new_ds, _, valid = d_update(self.critic, ds, self.real, self.fake, self.alpha)
        self.assertTrue(bool(valid))
        assert_tree_equal(self, saved_generator, self.generator)
        assert_tree_equal(self, saved_gs, gs)
        saved_critic, saved_ds = critic, new_ds
        generator, _, _, valid = g_update(self.generator, gs, critic, keys)
        self.assertTrue(bool(valid))
        assert_tree_equal(self, saved_critic, critic)
        assert_tree_equal(self, saved_ds, new_ds)
        self.assertFalse(bool(eqx.tree_equal(generator, self.generator)))
        self.assertFalse(bool(eqx.tree_equal(critic, self.critic)))

    def test_sampling_and_serialization(self) -> None:
        model = SDEGAN(self.config)
        model.fit(synthetic(5, 8), synthetic(6, 8), seed=0)
        paths = model.sample(5, seed=12)
        self.assertEqual(paths.shape, (5, 20, 4))
        self.assertEqual(paths.dtype, np.float32)
        self.assertTrue(np.isfinite(paths).all())
        self.assertGreater(float(paths[:, 0].std()), 0)
        np.testing.assert_array_equal(paths, model.sample(5, seed=12))
        self.assertFalse(np.array_equal(paths, model.sample(5, seed=13)))
        alternate = sample_paths(model.generator, 5, jr.PRNGKey(12), replace(self.config, sampling_batch_size=3))
        np.testing.assert_array_equal(paths, alternate)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                model.sample(invalid, seed=0)
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder)
            restored = SDEGAN.load(folder)
            np.testing.assert_array_equal(paths, restored.sample(5, seed=12))
            self.assertEqual(model.history, restored.history)
            self.assertEqual(model.config, restored.config)
            self.assertEqual(model.run_metadata, restored.run_metadata)
            self.assertIsNone(restored.training_result)
        model.fit(synthetic(5, 8), synthetic(6, 8), seed=0)
        np.testing.assert_array_equal(paths, model.sample(5, seed=12))


class TestTemporalPenalty(unittest.TestCase):
    def test_correlations_match_numpy_without_boundary_pairs(self) -> None:
        data = synthetic(17, 32)
        expected = np.array([
            [np.corrcoef(data[:, :-lag, j].ravel(), data[:, lag:, j].ravel())[0, 1]
             for j in range(4)] for lag in range(1, 6)
        ])
        np.testing.assert_allclose(temporal_autocorrelation(jnp.asarray(data)), expected, atol=1e-6)
        np.testing.assert_allclose(temporal_autocorrelation(jnp.asarray(data * 2 + 3)), expected, atol=1e-6)
        # Joining paths would add a spurious transition from -1 to +1.
        constants = jnp.broadcast_to(jnp.array([-1., 1.])[:, None, None], (2, 20, 4))
        np.testing.assert_allclose(temporal_autocorrelation(constants), 1, atol=1e-6)

    def test_penalty_and_degenerate_batch_gradients(self) -> None:
        data = jnp.asarray(synthetic(18, 128))
        target = temporal_autocorrelation(data)
        self.assertAlmostEqual(float(temporal_penalty(data, target)), 0)
        persistent = jnp.broadcast_to(data[:, :1], data.shape)
        self.assertGreater(float(temporal_penalty(persistent, target)), .5)
        for values in (data, jnp.zeros_like(data), jnp.ones_like(data)):
            loss, grads = jax.value_and_grad(temporal_penalty)(values, target)
            self.assertTrue(bool(jnp.isfinite(loss)))
            self.assertTrue(bool(jnp.isfinite(grads).all()))
        with self.assertRaises(ValueError):
            small_config(temporal_penalty_weight=-1)
        with self.assertRaises(ValueError):
            small_config(temporal_penalty_weight=np.inf)

    def test_active_objective_gradient_and_fixed_critic(self) -> None:
        config = small_config(temporal_penalty_weight=10)
        generator, critic = initialize_models(config, 21)
        keys = jr.split(jr.PRNGKey(22), config.batch_size)
        target = temporal_autocorrelation(jnp.asarray(synthetic(23, 128)))
        paths = generate_batch(generator, keys, config.generator_dt, config.brownian_tol)
        baseline = generator_loss(generator, critic, keys, replace(config, temporal_penalty_weight=0))
        active = generator_loss(generator, critic, keys, config, target)
        np.testing.assert_allclose(active, baseline + 10 * temporal_penalty(paths, target), rtol=1e-5)
        def penalty_only(g):
            return temporal_penalty(generate_batch(g, keys, config.generator_dt, config.brownian_tol), target)
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(penalty_only))(generator)
        self.assertTrue(bool(tree_finite((loss, grads))))
        self.assertGreater(float(optax.tree.norm(grads)), 0)
        # Moving in the penalty's negative gradient direction lowers that objective.
        moved = eqx.apply_updates(generator, jax.tree.map(lambda x: -1e-4 * x, grads))
        self.assertLess(float(penalty_only(moved)), float(loss))
        g_opt, d_opt, update, _ = make_updates(config, target)
        state = g_opt.init(eqx.filter(generator, eqx.is_inexact_array))
        saved_critic = critic
        _, _, metrics, valid = update(generator, state, critic, keys)
        self.assertTrue(bool(valid))
        self.assertEqual(metrics.shape, (6,))
        assert_tree_equal(self, saved_critic, critic)

    def test_training_target_and_serialization(self) -> None:
        config = small_config(temporal_penalty_weight=10)
        training = synthetic(24, 8)
        model = SDEGAN(config)
        history = model.fit(training, synthetic(25, 8), seed=0)
        np.testing.assert_allclose(model.run_metadata['temporal_penalty']['training_target'],
                                   temporal_autocorrelation(jnp.asarray(training)), atol=1e-6)
        self.assertGreater(history[1]['temporal_penalty'], 0)
        self.assertAlmostEqual(history[1]['generator_loss'],
                               history[1]['generator_adversarial_loss']
                               + 10 * history[1]['temporal_penalty'], places=5)
        self.assertTrue(np.isfinite(history[1]['validation_temporal_error']))
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = SDEGAN.load(directory)
            self.assertEqual(restored.config.temporal_penalty_weight, 10)
            self.assertEqual(restored.run_metadata, model.run_metadata)
            self.assertEqual(restored.history, model.history)
            np.testing.assert_array_equal(model.sample(5, 26), restored.sample(5, 26))


class TestAbsoluteTemporalPenalty(unittest.TestCase):
    def test_original_log_units_and_constant_gradients(self) -> None:
        data = synthetic(60, 128)
        mean = np.array([.01, -.02, .005, .03], dtype=np.float32)
        std = np.array([.02, .04, .015, .03], dtype=np.float32)
        raw = data * std + mean
        expected = np.array([
            [np.corrcoef(np.abs(raw[:, :-lag, j]).ravel(),
                         np.abs(raw[:, lag:, j]).ravel())[0, 1]
             for j in range(4)] for lag in range(1, 6)
        ])
        target = temporal_autocorrelation(jnp.abs(jnp.asarray(data) + mean / std))
        np.testing.assert_allclose(target, expected, atol=1e-6)
        reference = (target, jnp.asarray(mean / std))
        self.assertAlmostEqual(float(absolute_temporal_penalty(jnp.asarray(data), reference)), 0)
        for values in (data, np.zeros_like(data), np.ones_like(data)):
            value, grads = jax.value_and_grad(absolute_temporal_penalty)(jnp.asarray(values), reference)
            self.assertTrue(bool(jnp.isfinite(value) & jnp.isfinite(grads).all()))
        for weight in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                small_config(absolute_temporal_penalty_weight=weight)

    def test_generator_objective_and_update(self) -> None:
        config = small_config(generator_output='increments', absolute_temporal_penalty_weight=1)
        generator, critic = initialize_models(config, 61)
        keys = jr.split(jr.PRNGKey(62), config.batch_size)
        offset = jnp.array([.1, -.2, .05, .3])
        target = temporal_autocorrelation(jnp.abs(synthetic(63, 32) + offset))
        reference = (target, offset)
        baseline = generator_loss(generator, critic, keys, replace(config, absolute_temporal_penalty_weight=0))
        active = generator_loss(generator, critic, keys, config, absolute_temporal_reference=reference)
        paths = generate_batch(generator, keys, config.generator_dt, config.brownian_tol)
        np.testing.assert_allclose(active, baseline + absolute_temporal_penalty(paths, reference), rtol=1e-5)

        def penalty(g):
            return absolute_temporal_penalty(
                generate_batch(g, keys, config.generator_dt, config.brownian_tol), reference)

        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(penalty))(generator)
        self.assertTrue(bool(tree_finite((value, grads))))
        norm = optax.tree.norm(grads)
        self.assertGreater(float(norm), 0)
        moved = eqx.apply_updates(generator, jax.tree.map(lambda x: -1e-5 * x / norm, grads))
        self.assertLess(float(penalty(moved)), float(value))
        g_opt, _, update, _ = make_updates(config, absolute_temporal_reference=reference)
        state = g_opt.init(eqx.filter(generator, eqx.is_inexact_array))
        _, _, metrics, valid = update(generator, state, critic, keys)
        self.assertTrue(bool(valid))
        self.assertEqual(metrics.shape, (6,))
        np.testing.assert_allclose(metrics[-1], absolute_temporal_penalty(paths, reference), rtol=1e-5)

    def test_training_reference_logging_and_restore(self) -> None:
        config = small_config(generator_output='increments', absolute_temporal_penalty_weight=1,
                              retain_validation_checkpoints=True, generator_steps=2, validation_every=1)
        data = synthetic(64, 16)
        offset = np.array([.03, -.02, .01, .04], dtype=np.float32)
        model = SDEGAN(config)
        model.fit(data, synthetic(65, 8), seed=0, absolute_return_offset=offset)
        reference = model.run_metadata['absolute_temporal_penalty']['training_reference']
        np.testing.assert_allclose(reference['training_target'],
                                   temporal_autocorrelation(jnp.abs(data + offset)), atol=1e-6)
        np.testing.assert_array_equal(reference['return_offset'], offset)
        self.assertTrue(all(np.isfinite(row['validation_absolute_temporal_error']) for row in model.history))
        row = model.history[1]
        self.assertAlmostEqual(row['generator_loss'], row['generator_adversarial_loss']
                               + row['absolute_temporal_penalty'], places=5)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = SDEGAN.load(directory)
            self.assertEqual(restored.run_metadata, model.run_metadata)
            self.assertEqual(restored.config, model.config)
            np.testing.assert_array_equal(restored.sample(5, 66), model.sample(5, 66))
            snapshot = SDEGAN.load(Path(directory) / 'checkpoints/step_0002')
            self.assertTrue(np.isfinite(snapshot.run_metadata['validation_absolute_temporal_error']))
        with self.assertRaisesRegex(ValueError, 'absolute_return_offset'):
            SDEGAN(config).fit(data, synthetic(65, 8))
        for bad in ([0., 1.], [0., 0., np.nan, 0.]):
            with self.assertRaisesRegex(ValueError, 'absolute_return_offset'):
                SDEGAN(config).fit(data, synthetic(65, 8), absolute_return_offset=bad)


class TestValidationCheckpoints(unittest.TestCase):
    def test_relative_volatility_error(self) -> None:
        real = synthetic(30, 8)
        self.assertAlmostEqual(volatility_error(real, real), 0)
        self.assertAlmostEqual(volatility_error(real, 2 * real), 1)
        self.assertAlmostEqual(volatility_error(real, .5 * real), .5)
        self.assertEqual(volatility_error(np.zeros_like(real), np.zeros_like(real)), 0)
        with self.assertRaises(ValueError):
            small_config(retain_validation_checkpoints=1)

    def test_every_validation_snapshot_restores(self) -> None:
        config = small_config(generator_steps=3, validation_every=2,
                              temporal_penalty_weight=30, retain_validation_checkpoints=True)
        model = SDEGAN(config)
        model.fit(synthetic(31, 8), synthetic(32, 8), seed=0)
        snapshots = model.training_result['validation_checkpoints']
        self.assertEqual([s['step'] for s in snapshots], [0, 2, 3])
        scores = [s['validation_mmd2'] for s in snapshots]
        self.assertEqual(model.run_metadata['selected_step'], snapshots[int(np.argmin(scores))]['step'])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model.save(folder)
            with (folder / 'validation_checkpoints.csv').open() as handle:
                index = list(csv.DictReader(handle))
            self.assertEqual(len(index), 3)
            for snapshot, row in zip(snapshots, index):
                restored = SDEGAN.load(folder / row['path'])
                self.assertEqual(restored.run_metadata['selected_step'], snapshot['step'])
                self.assertEqual(len(restored.history), snapshot['step'] + 1)
                self.assertEqual(restored.run_metadata['validation_score'], snapshot['validation_mmd2'])
                self.assertTrue(np.isfinite(float(row['validation_volatility_error'])))
                expected = sample_paths(snapshot['generator'], 5, jr.PRNGKey(33), config)
                np.testing.assert_array_equal(restored.sample(5, 33), expected)
                self.assertFalse((folder / row['path'] / 'validation_checkpoints.csv').exists())
        baseline = SDEGAN(small_config())
        baseline.fit(synthetic(31, 8), synthetic(32, 8), seed=0)
        self.assertEqual(baseline.training_result['validation_checkpoints'], [])


class TestSignatureLoss(unittest.TestCase):
    def test_features_against_piecewise_linear_integrals(self) -> None:
        data = synthetic(40, 5) * .2
        cumulative = np.concatenate([np.zeros((5, 1, 4)), data.cumsum(axis=1)], axis=1)
        repeated = np.repeat(cumulative, 2, axis=1)
        time = np.broadcast_to(np.linspace(0, 1, 41)[None, :, None], (5, 41, 1))
        path = np.concatenate([time, repeated[:, 1:], repeated[:, :-1]], axis=2)
        increments = np.diff(path, axis=1)
        midpoints = (path[:, :-1] + path[:, 1:]) / 2 - path[:, :1]
        second = np.einsum('nti,ntj->nij', midpoints, increments)
        expected = np.concatenate([path[:, -1] - path[:, 0], second.reshape(5, -1)], axis=1)
        np.testing.assert_allclose(signature_features(jnp.asarray(data)), expected, rtol=1e-4, atol=2e-5)
        reference = fit_signature_reference(jnp.asarray(data))
        self.assertEqual(int(reference[2].sum()), 88)
        self.assertEqual(float(reference[2][0]), 0)
        self.assertEqual(float(reference[2][9]), 0)
        self.assertAlmostEqual(float(signature_error(jnp.asarray(data), reference)), 0, places=5)

    def test_unbiased_estimate_and_constant_path_gradients(self) -> None:
        data = jnp.asarray(synthetic(41, 8) * .2)
        reference = fit_signature_reference(jnp.asarray(synthetic(42, 32) * .2))
        mean, scale, active = map(np.asarray, reference)
        delta = (np.asarray(signature_features(data)) - mean) / scale * active
        expected = np.mean([np.dot(delta[i], delta[j]) / active.sum()
                            for i in range(8) for j in range(8) if i != j])
        actual = float(signature_loss(data, reference))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        naive = np.sum(delta.mean(axis=0) ** 2) / active.sum()
        variance_bias = np.sum(delta.var(axis=0, ddof=1)) / (8 * active.sum())
        np.testing.assert_allclose(naive - actual, variance_bias, rtol=1e-4, atol=1e-6)
        pair = data[:2]
        centered_reference = (signature_features(pair).mean(axis=0), reference[1], reference[2])
        self.assertLess(float(signature_loss(pair, centered_reference)), 0)
        for values in (data, jnp.zeros_like(data)):
            value, gradient = jax.value_and_grad(signature_loss)(values, reference)
            self.assertTrue(bool(jnp.isfinite(value) & jnp.isfinite(gradient).all()))
        for kwargs in ({'signature_loss_weight': -1}, {'signature_loss_weight': np.nan},
                       {'signature_loss_weight': 1, 'batch_size': 1}):
            with self.assertRaises(ValueError):
                small_config(**kwargs)

    def test_signature_gradient_through_generator(self) -> None:
        config = small_config(temporal_penalty_weight=10, signature_loss_weight=1)
        generator, critic = initialize_models(config, 43)
        keys = jr.split(jr.PRNGKey(44), 2)
        data = jnp.asarray(synthetic(45, 32))
        temporal = temporal_autocorrelation(data)
        reference = fit_signature_reference(data)
        baseline = generator_loss(generator, critic, keys, replace(config, signature_loss_weight=0), temporal)
        active = generator_loss(generator, critic, keys, config, temporal, reference)
        paths = generate_batch(generator, keys, config.generator_dt, config.brownian_tol)
        np.testing.assert_allclose(active, baseline + signature_loss(paths, reference), rtol=1e-5)
        def loss(g):
            return signature_loss(generate_batch(g, keys, config.generator_dt, config.brownian_tol), reference)
        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(generator)
        norm = optax.tree.norm(grads)
        self.assertTrue(bool(tree_finite((value, grads))))
        self.assertGreater(float(norm), 0)
        moved = eqx.apply_updates(generator, jax.tree.map(lambda x: -1e-4 * x / norm, grads))
        self.assertLess(float(loss(moved)), float(value))
        g_opt, _, update, _ = make_updates(config, temporal, reference)
        state = g_opt.init(eqx.filter(generator, eqx.is_inexact_array))
        saved_critic = critic
        _, _, metrics, valid = update(generator, state, critic, keys)
        self.assertTrue(bool(valid))
        self.assertEqual(metrics.shape, (6,))
        assert_tree_equal(self, saved_critic, critic)

    def test_reference_logging_and_checkpoint_roundtrip(self) -> None:
        config = small_config(generator_steps=2, validation_every=1, temporal_penalty_weight=10,
                              signature_loss_weight=1, retain_validation_checkpoints=True)
        data = synthetic(46, 16)
        model = SDEGAN(config)
        history = model.fit(data, synthetic(47, 8), seed=0)
        reference = model.run_metadata['signature_loss']['training_reference']
        for key, expected in zip(('mean', 'scale', 'active'), fit_signature_reference(jnp.asarray(data))):
            np.testing.assert_allclose(reference[key], expected)
        self.assertTrue(all(np.isfinite(row['validation_signature_error']) for row in history))
        self.assertAlmostEqual(history[1]['generator_loss'], history[1]['generator_adversarial_loss']
                               + 10 * history[1]['temporal_penalty'] + history[1]['signature_loss'], places=5)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            for path in (Path(directory), Path(directory) / 'checkpoints/step_0002'):
                restored = SDEGAN.load(path)
                self.assertEqual(restored.config.signature_loss_weight, 1)
                self.assertEqual(restored.run_metadata['signature_loss'], model.run_metadata['signature_loss'])
                self.assertTrue(np.isfinite(restored.history[-1]['validation_signature_error']))
            np.testing.assert_array_equal(model.sample(5, 48), SDEGAN.load(directory).sample(5, 48))


class TestSmokeRun(unittest.TestCase):
    def test_50_baseline_cycles(self) -> None:
        config = replace(SDEGANConfig(), generator_steps=50, validation_every=10)
        model = SDEGAN(config)
        history = model.fit(synthetic(1000, 128), synthetic(1001, 64), seed=0)
        self.assertEqual(len(history), 51)
        self.assertEqual(len([r for r in history if r['validation_mmd2'] is not None]), 6)
        self.assertTrue(all(np.isfinite(v) for row in history for v in row.values() if v is not None))
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder)
            np.testing.assert_array_equal(model.sample(17, seed=42), SDEGAN.load(folder).sample(17, seed=42))
        # Paired ensemble comparison uses identical initial noise and Brownian trees.
        key = jr.PRNGKey(900)
        coarse = sample_paths(model.generator, 256, key, config)
        fine = sample_paths(model.generator, 256, key, replace(config, generator_dt=.125))
        differences = (coarse - fine).mean(axis=1)
        paired_se = differences.std(axis=0, ddof=1) / np.sqrt(len(differences))
        # Report sensitivity; this short smoke run is not a solver-convergence claim.
        report = {
            'cycles': 50, 'batch_size': config.batch_size,
            'median_cycle_seconds_after_compilation': float(np.median([r['cycle_seconds'] for r in history[2:]])),
            'elapsed_seconds': history[-1]['elapsed_seconds'],
            'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'selected_step': model.run_metadata['selected_step'],
            'validation_mmd2': model.run_metadata['validation_score'],
            'coarse_minus_fine_mean_by_stock': differences.mean(axis=0).tolist(),
            'paired_standard_error_by_stock': paired_se.tolist(),
            'coarse_volatility_by_stock': coarse.std(axis=(0, 1)).tolist(),
            'fine_volatility_by_stock': fine.std(axis=(0, 1)).tolist(),
        }
        print('\nSMOKE_REPORT ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    unittest.main()
