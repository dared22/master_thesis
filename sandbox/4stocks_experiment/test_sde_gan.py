"""Run with sandbox/.venv/bin/python -m unittest from this directory.

TestSmokeRun performs the full 50-cycle CPU acceptance run. The remaining
classes provide faster numerical, gradient, and API regression checks.
"""

from dataclasses import replace
import json
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

from architecture import solve_sde
from sde_gan import SDEGAN, SDEGANConfig
from training import (
    generate_batch, generator_loss, gradient_penalty, initialize_models,
    kernel_bandwidth, make_updates, mmd_squared, sample_paths, train,
    tree_finite, validate_windows,
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
            return model, state, jnp.ones(2), jnp.array(True)
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
            return model, state, jnp.ones(2), jnp.array(True)
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
