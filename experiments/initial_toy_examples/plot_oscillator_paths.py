"""Simulate ten randomly forced oscillators with explicit Euler--Maruyama."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path(__file__).resolve().parents[2] / "thesis/figures/theory/oscillator-paths.pdf"
rng = np.random.default_rng(42)
h, duration, omega, sigma = 0.001, 20.0, 1.0, 0.3
t = np.linspace(0, duration, round(duration / h) + 1)
x = np.ones((10, len(t)))
v = np.zeros_like(x)
for i in range(len(t) - 1):
    x[:, i + 1] = x[:, i] + h * v[:, i]
    v[:, i + 1] = v[:, i] - omega**2 * h * x[:, i] + sigma * np.sqrt(h) * rng.normal(size=10)
assert np.isfinite(x).all() and np.isfinite(v).all()
np.testing.assert_allclose(np.diff(x, axis=1), h * v[:, :-1], atol=1e-14)
# With noise removed, Euler's oscillator approaches cos(t), -sin(t).
mean_state = np.array([1.0, 0.0])
for _ in range(len(t) - 1):
    mean_state = np.array([mean_state[0] + h * mean_state[1],
                           mean_state[1] - omega**2 * h * mean_state[0]])
exact_state = np.array([np.cos(duration), -np.sin(duration)])
assert np.linalg.norm(mean_state - exact_state) < 0.02

fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
for i, path in enumerate(x):
    ax.plot(t, path, color="#0072B2", linewidth=0.8, alpha=0.6,
            label="10 Euler–Maruyama paths" if i == 0 else None)
ax.plot(t, np.cos(t), color="black", linewidth=1.5, linestyle="--",
        label=r"Without random forcing: $\cos(t)$")
ax.set(xlabel=r"Time $t$", ylabel=r"Position $X_t$", xlim=(0, duration))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.15)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.savefig(output)
fig.savefig("/tmp/oscillator-paths-preview.png", dpi=180)
plt.close(fig)
print(f"Saved {output}; 10 paths, seed 42, h={h}, duration={duration}.")
print(f"Deterministic endpoint error: {np.linalg.norm(mean_state - exact_state):.6f}")
