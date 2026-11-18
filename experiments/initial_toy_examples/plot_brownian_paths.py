"""Illustrate ten standard Brownian paths and their pointwise distribution."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path(__file__).resolve().parents[2] / "thesis/figures/theory/brownian-paths.pdf"
rng = np.random.default_rng(42)
t = np.linspace(0, 1, 1001)
increments = rng.normal(size=(10, len(t) - 1)) * np.sqrt(np.diff(t))
paths = np.column_stack((np.zeros(10), np.cumsum(increments, axis=1)))
bound = 1.96 * np.sqrt(t)
assert paths.shape == (10, len(t))
assert np.all(paths[:, 0] == 0)
np.testing.assert_allclose(np.diff(paths, axis=1), increments, atol=1e-14)

fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
ax.fill_between(t, -bound, bound, color="0.85", alpha=0.7,
                label=r"Pointwise 95% bounds: $\pm1.96\sqrt{t}$")
ax.plot(t, bound, color="0.5", linewidth=0.8, linestyle=":")
ax.plot(t, -bound, color="0.5", linewidth=0.8, linestyle=":")
for i, path in enumerate(paths):
    ax.plot(t, path, color="#0072B2", linewidth=0.7, alpha=0.65,
            label="10 sample paths" if i == 0 else None)
ax.axhline(0, color="black", linewidth=1.5, linestyle="--",
           label=r"Mean: $\mathbb{E}[W_t]=0$")
ax.set(xlabel=r"Time $t$", ylabel=r"$W_t$", xlim=(0, 1))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.15)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.savefig(output)
fig.savefig("/tmp/brownian-paths-preview.png", dpi=180)
plt.close(fig)
print(f"Saved {output}; 10 paths, seed 42, 1000 increments each.")
