"""Reproduce a recorded toy-model fit and plot 50 new learned SDE paths."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Reuse the notebook's definitions without running its experiment sweeps.
folder = Path(__file__).resolve().parent
notebook_path = folder / "sde_test_data.ipynb"
notebook = json.loads(notebook_path.read_text())
for index in (2, 4, 6, 8, 10, 16, 20):
    exec("".join(notebook["cells"][index]["source"]), globals())
from train_sde import train_nsde

artifacts = folder / "experiment_results/neural_sde_paths"
artifacts.mkdir(exist_ok=True)
checkpoint = artifacts / "model.eqx"
metadata_path = artifacts / "results.json"
reference = next(
    row for row in csv.DictReader(
        (folder / "experiment_results/sde_test_data/repeated_sweeps.csv").open()
    )
    if row["sweep"] == "observation_density" and row["data_seed"] == "1000"
    and row["n_observations"] == "15000"
)
if checkpoint.exists():
    model = eqx.tree_deserialise_leaves(checkpoint, NeuralSDE(jr.PRNGKey(0)))
    metadata = json.loads(metadata_path.read_text())
    print("Loaded reproduced checkpoint.", flush=True)
else:
    print("Simulating 50 training paths and the original validation paths...", flush=True)
    training_keys = jr.split(jr.PRNGKey(1000), 50)
    original_keys = jr.split(jr.PRNGKey(42), 100)
    training_paths = jax.jit(jax.vmap(simulate_one))(training_keys)
    validation_paths = jax.jit(jax.vmap(simulate_one))(original_keys[40:])
    jax.block_until_ready(validation_paths)
    print("Fitting the full-observation model (training seed 2000)...", flush=True)
    fit = train_nsde(
        training_paths, ts, validation_paths, ts,
        model_factory=NeuralSDE, loss_function=loss_fn,
        seed=2000, validation_seed=102, verbose=True,
    )
    model = fit["model"]
    grid = jnp.linspace(-np.pi, np.pi, 512, endpoint=False)
    true_b = -jnp.sin(grid)
    fitted_b = jax.vmap(model.b)(grid)
    drift_error = 100 * float(jnp.sqrt(jnp.mean((fitted_b - true_b)**2))) / float(
        jnp.sqrt(jnp.mean(true_b**2))
    )
    fitted_sigma = float(model.s(0.0))
    # Confirm this is the same recorded fit, within numerical precision.
    np.testing.assert_allclose(
        [drift_error, fitted_sigma, fit["best_validation_loss"]],
        [float(reference["drift_error_percent"]), float(reference["sigma_estimate"]),
         float(reference["validation_loss"])], rtol=1e-4, atol=1e-6,
    )
    assert fit["best_step"] == int(reference["best_step"])
    metadata = {
        "training_data_seed": 1000, "training_seed": 2000,
        "validation_seed": 102, "simulation_seed": 2026,
        "n_training_paths": 50, "n_generated_paths": 50,
        "duration": float(t1), "n_saved_times": len(ts),
        "integration_step": float(solver_ts[1] - solver_ts[0]),
        "best_step": fit["best_step"], "sigma_estimate": fitted_sigma,
        "drift_error_percent": drift_error,
        "validation_loss": fit["best_validation_loss"],
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    }
    eqx.tree_serialise_leaves(checkpoint, model)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    del training_paths, validation_paths, fit
    print("Verified fit against the recorded experiment and saved weights.", flush=True)

@eqx.filter_jit
def generate_paths(model, key):
    """Euler--Maruyama on the same grid, with new independent normal draws."""
    def step(state, step_key):
        h = solver_ts[1] - solver_ts[0]
        new_state = (state + h * jax.vmap(model.b)(state)
                     + model.s(0.0) * jnp.sqrt(h) * jr.normal(step_key, (50,)))
        return new_state, new_state

    keys = jr.split(key, len(solver_ts) - 1)
    _, states = jax.lax.scan(step, jnp.zeros(50), keys)
    return jnp.concatenate((jnp.zeros((1, 50)), states), axis=0)[::steps_per_observation].T

paths = np.asarray(generate_paths(model, jr.PRNGKey(2026)))
assert paths.shape == (50, len(ts)) and np.isfinite(paths).all()
assert np.all(paths[:, 0] == 0)
np.savez_compressed(artifacts / "paths.npz", times=np.asarray(ts), paths=paths)
fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
for path in paths:
    ax.plot(np.asarray(ts), path, color="#0072B2", linewidth=0.55, alpha=0.35,
            rasterized=True)
ax.set(xlabel=r"Time $t$", ylabel=r"State $X_t$", xlim=(0, t1))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.15)
output = folder.parents[1] / "thesis/figures/theory/neural-sde-paths.pdf"
fig.savefig(output, dpi=300)
fig.savefig("/tmp/neural-sde-paths-preview.png", dpi=180)
plt.close(fig)
print(f"Saved {output}: 50 new paths from the fitted neural SDE.", flush=True)
