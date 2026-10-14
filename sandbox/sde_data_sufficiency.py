"""Repeated SDE data-size experiments and four views of the same results."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def increments(paths: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Flatten consecutive increments using their actual, possibly unequal times."""
    paths, times = np.asarray(paths), np.asarray(times)
    if paths.ndim != 2 or times.ndim != 1 or paths.shape[1] != times.size:
        raise ValueError("Paths must have shape (n_paths, len(times)).")
    if paths.shape[0] < 1 or times.size < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Require nonempty paths and strictly increasing times.")
    if not np.isfinite(paths).all() or not np.isfinite(times).all():
        raise ValueError("Paths and times must be finite.")
    starts = paths[:, :-1]
    return (
        starts.reshape(-1), np.diff(paths, axis=1).reshape(-1),
        np.broadcast_to(np.diff(times), starts.shape).reshape(-1),
    )


def _errors(
    fitted_b: np.ndarray, fitted_sigma: float, true_b: np.ndarray,
    sigma: float, drift_tolerance: float, diffusion_tolerance: float,
) -> dict[str, float]:
    drift_error = float(np.sqrt(np.mean((fitted_b - true_b) ** 2))
                        / np.sqrt(np.mean(true_b ** 2)))
    diffusion_error = abs(float(fitted_sigma) - sigma) / sigma
    return {
        "drift_error": drift_error,
        "diffusion_error": diffusion_error,
        "combined_error": max(drift_error / drift_tolerance,
                              diffusion_error / diffusion_tolerance),
    }


def save_results(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Atomically checkpoint completed fits to CSV."""
    if not rows:
        raise ValueError("No results to save.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_sweep(
    path_pools: Sequence[np.ndarray], times: np.ndarray,
    test_paths: np.ndarray, trajectory_counts: Sequence[int],
    observation_counts: Sequence[int], *,
    model_factory: Callable[..., Any], optimizer_init: Callable[..., Any],
    train_step: Callable[..., Any], evaluate_loss: Callable[..., Any],
    amplitude: float, wave_number: float, sigma: float,
    training_steps: int = 1500, batch_size: int = 2048,
    test_size: int = 16384, seed: int = 2026,
    drift_tolerance: float = 0.05, diffusion_tolerance: float = 0.02,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fit fresh neural/direct models to nested subsets of independent pools.

    Each pool is one data replicate, disjoint from the fixed test paths.
    All fits use the same update budget, not a guarantee of convergence.
    Test data only score completed fits; they never select checkpoints.
    """
    times = np.asarray(times)
    counts_n = sorted(set(trajectory_counts))
    counts_m = sorted(set(observation_counts))
    if (not counts_n or not counts_m or not len(path_pools)
            or any(int(v) != v for v in counts_n + counts_m)
            or min(counts_n) < 1 or min(counts_m) < 2
            or max(counts_m) > times.size):
        raise ValueError("Require valid integer path/observation counts and pools.")
    if (sigma <= 0 or amplitude == 0 or wave_number == 0
            or drift_tolerance <= 0 or diffusion_tolerance <= 0
            or min(training_steps, batch_size, test_size) < 1):
        raise ValueError("Require positive budgets/tolerances/sigma and nonzero drift.")
    counts_n, counts_m = [int(v) for v in counts_n], [int(v) for v in counts_m]
    pools = [np.asarray(pool) for pool in path_pools]
    for pool in pools:
        increments(pool, times)
        if pool.shape[0] < max(counts_n):
            raise ValueError("Every data pool must contain enough training paths.")
    test = increments(test_paths, times)
    test_rng = np.random.default_rng(seed)
    test_indices = test_rng.choice(test[0].size, min(test_size, test[0].size),
                                   replace=False)
    test_batch = tuple(jnp.asarray(value[test_indices]) for value in test)
    grid = jnp.linspace(-np.pi / abs(wave_number), np.pi / abs(wave_number),
                        512, endpoint=False)
    true_b = np.asarray(-amplitude * wave_number * jnp.sin(wave_number * grid))
    rows: list[dict[str, Any]] = []
    total = len(pools) * len(counts_n) * len(counts_m)
    completed = 0
    for replicate, pool in enumerate(pools):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        ordered_pool = pool[rng.permutation(pool.shape[0])]
        # Shared initialization and batch RNG within a replicate reduce noise.
        model_key = jr.fold_in(jr.PRNGKey(seed), replicate)
        for n in counts_n:
            for m in counts_m:
                started = perf_counter()
                selected = np.rint(np.linspace(0, times.size - 1, m)).astype(int)
                data = increments(ordered_pool[:n, selected], times[selected])
                data_jax = tuple(jnp.asarray(value) for value in data)
                fit = model_factory(model_key)
                state = optimizer_init(fit)
                batch_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, replicate, 1]))
                diagnostic_indices = batch_rng.integers(0, data[0].size,
                                                        size=batch_size)
                diagnostic_batch = tuple(v[diagnostic_indices] for v in data_jax)
                diagnostic = []
                for step in range(training_steps):
                    indices = batch_rng.integers(0, data[0].size, size=batch_size)
                    fit, state, loss = train_step(
                        fit, state, *(v[indices] for v in data_jax))
                    if (step + 1) % 100 == 0 or step == training_steps - 1:
                        value = float(evaluate_loss(fit, *diagnostic_batch))
                        if not np.isfinite(value):
                            raise FloatingPointError(f"Nonfinite loss at n={n}, m={m}.")
                        diagnostic.append(value)
                fitted_b = np.asarray(jax.vmap(fit.b)(grid))
                fitted_sigma = float(fit.s(grid[0]))
                neural_errors = _errors(fitted_b, fitted_sigma, true_b, sigma,
                                        drift_tolerance, diffusion_tolerance)
                neural_nll = float(evaluate_loss(fit, *test_batch))
                neural_seconds = perf_counter() - started
                direct_started = perf_counter()
                x, dx, delta = (v.astype(np.float64) for v in data)
                basis = -wave_number * np.sin(wave_number * x)
                denominator = np.sum(basis ** 2 * delta)
                if denominator <= 0:
                    raise ValueError("Direct drift estimate has no identifiable basis.")
                direct_a = float(np.sum(basis * dx) / denominator)
                residual = dx - direct_a * basis * delta
                direct_sigma = float(np.sqrt(np.mean(residual ** 2 / delta)))
                if direct_sigma <= 0:
                    raise ValueError("Direct diffusion estimate is nonpositive.")
                direct_errors = _errors(
                    true_b * direct_a / amplitude, direct_sigma, true_b, sigma,
                    drift_tolerance, diffusion_tolerance)
                tx, tdx, th = (np.asarray(v) for v in test_batch)
                direct_residual = tdx + direct_a * wave_number * np.sin(
                    wave_number * tx) * th
                direct_nll = float(np.mean(
                    0.5 * direct_residual ** 2 / (direct_sigma ** 2 * th)
                    + np.log(direct_sigma)))
                direct_seconds = perf_counter() - direct_started
                common = {
                    "replicate": replicate, "seed": seed, "n_train": n,
                    "n_observations": m, "n_increments": n * (m - 1),
                    "duration": float(times[-1] - times[0]),
                    "mean_spacing": float(np.mean(np.diff(times[selected]))),
                    "drift_tolerance": drift_tolerance,
                    "diffusion_tolerance": diffusion_tolerance,
                }
                if not np.isfinite([*neural_errors.values(), neural_nll]).all():
                    raise FloatingPointError("Nonfinite fitted coefficients or scores.")
                for method, errors, nll, a_hat, s_hat in (
                    ("neural", neural_errors, neural_nll, float("nan"), fitted_sigma),
                    ("direct", direct_errors, direct_nll, direct_a, direct_sigma),
                ):
                    rows.append({
                        **common, "method": method, **errors, "test_nll": nll,
                        "amplitude_estimate": a_hat, "sigma_estimate": s_hat,
                        "training_steps": training_steps if method == "neural" else 0,
                        "batch_size": batch_size if method == "neural" else 0,
                        "last_train_loss": diagnostic[-1] if method == "neural" else float("nan"),
                        "last_diagnostic_change": (diagnostic[-1] - diagnostic[-2]
                            if method == "neural" and len(diagnostic) > 1 else float("nan")),
                        "fit_seconds": neural_seconds if method == "neural" else direct_seconds,
                    })
                if output_csv is not None:
                    save_results(rows, output_csv)
                completed += 1
                print(f"Fit {completed}/{total}: replicate={replicate}, n={n}, m={m}, "
                      f"neural C={neural_errors['combined_error']:.3f}, "
                      f"direct C={direct_errors['combined_error']:.3f}", flush=True)
    return rows


def plot_sweep(
    rows: list[dict[str, Any]], *, method: str = "neural",
    metric: str = "combined_error", output_path: str | Path | None = None,
) -> Figure:
    """Show heatmap, learning curves, surface, and budget view in one figure.

    Cells/curves show medians; bands show the 25–75% replicate range, not CIs.
    Heatmap stars indicate that the 90th percentile combined error passes.
    The budget plot compares observed allocations, not optimized budgets.
    """
    labels = {"combined_error": "Combined error / tolerance",
              "drift_error": "Relative drift RMSE",
              "diffusion_error": "Relative diffusion error",
              "test_nll": "Fixed-test increment NLL"}
    if metric not in labels:
        raise ValueError(f"Choose a metric from {list(labels)}.")
    data = [row for row in rows if row["method"] == method]
    if not data:
        raise ValueError(f"No rows for method {method!r}.")
    ns = sorted({int(row["n_train"]) for row in data})
    ms = sorted({int(row["n_observations"]) for row in data})
    if min(len(ns), len(ms)) < 2:
        raise ValueError("Four-view plotting requires at least a 2-by-2 sweep.")
    median = np.empty((len(ms), len(ns)))
    low, high, upper_c = (np.empty_like(median) for _ in range(3))
    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            group = [row for row in data if row["n_train"] == n
                     and row["n_observations"] == m]
            if not group:
                raise ValueError("Plotting requires a complete rectangular grid.")
            values = [float(row[metric]) for row in group]
            if not np.isfinite(values).all():
                raise ValueError("Plot scores must be finite.")
            low[i, j], median[i, j], high[i, j] = np.quantile(values, [.25, .5, .75])
            upper_c[i, j] = np.quantile([float(row["combined_error"])
                                        for row in group], .9)
    fig = plt.figure(figsize=(16, 11))
    fig.subplots_adjust(left=.08, right=.95, bottom=.08, top=.87,
                        wspace=.30, hspace=.35)
    heat = fig.add_subplot(2, 2, 1)
    curves = fig.add_subplot(2, 2, 2)
    surface = fig.add_subplot(2, 2, 3, projection="3d")
    budget = fig.add_subplot(2, 2, 4)
    # Log coordinates preserve uneven data-size spacing in cells and surface.
    log_n, log_m = np.log10(ns), np.log10(ms)
    mesh = heat.pcolormesh(log_n, log_m, median, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=heat, label=labels[metric])
    if upper_c.min() < 1 < upper_c.max():
        boundary = heat.contour(log_n, log_m, upper_c, levels=[1],
                                colors="white", linewidths=1.5)
        heat.clabel(boundary, fmt={1: "90% quantile C=1"}, fontsize=8)
    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            heat.text(log_n[j], log_m[i], f"{median[i, j]:.3g}"
                      + (" *" if upper_c[i, j] <= 1 else ""),
                      ha="center", va="center", color="white", fontsize=8)
    heat.set(xticks=log_n, xticklabels=ns, yticks=log_m, yticklabels=ms,
             xlabel="Training trajectories (log spacing)",
             ylabel="Observations per trajectory (log spacing)",
             title="1. Heatmap; * meets both tolerances at q90")
    colors = plt.get_cmap("plasma")(np.linspace(.1, .85, len(ms)))
    for i, (m, color) in enumerate(zip(ms, colors)):
        curves.plot(ns, median[i], "o-", color=color, label=f"m={m:,}")
        curves.fill_between(ns, low[i], high[i], color=color, alpha=.18)
        increments_budget = np.asarray(ns) * (m - 1)
        budget.plot(increments_budget, median[i], "o-", color=color, label=f"m={m:,}")
        budget.fill_between(increments_budget, low[i], high[i], color=color, alpha=.18)
    for ax, xlabel, title in (
        (curves, "Training trajectories", "2. Learning curves; bands = replicate IQR"),
        (budget, "Total training increments: n × (m − 1)", "4. Budget vs. allocation"),
    ):
        ax.set(xscale="log", xlabel=xlabel, ylabel=labels[metric], title=title)
        if metric == "combined_error":
            ax.axhline(1, color="black", linestyle="--", linewidth=1,
                       label="Both accuracy tolerances")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    xx, yy = np.meshgrid(log_n, log_m)
    surface.plot_surface(xx, yy, median, cmap="viridis", alpha=.8,
                         vmin=float(median.min()), vmax=float(median.max()))
    surface.scatter(xx, yy, median, color="black", s=10, depthshade=False)
    surface.set(xticks=log_n, xticklabels=ns, yticks=log_m, yticklabels=ms,
                xlabel="Training trajectories", ylabel="Observations",
                zlabel=labels[metric], title="3. Surface; dots = measured cells (log axes)")
    surface.view_init(elev=25, azim=-125)
    repeats = len({row["replicate"] for row in data})
    fig.suptitle(f"{method.capitalize()} SDE data sufficiency — {repeats} data replicates\n"
                 "Fixed horizon and optimizer budget; median scores", fontsize=14)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    return fig
