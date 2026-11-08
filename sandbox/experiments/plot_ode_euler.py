"""Plot the exact solution and Euler approximations of x' = 2x, x(0) = 1."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path(__file__).resolve().parents[2] / "thesis/figures/theory/ode-euler.pdf"
t = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(6.2, 3.2), layout="constrained")
ax.plot(t, np.exp(2 * t), color="black", linewidth=2,
        label=r"Exact: $x(t)=e^{2t}$")
errors = []
for h, color, marker in [(0.5, "#D55E00", "s"),
                         (0.2, "#0072B2", "^"),
                         (0.05, "#009E73", "o")]:
    times = np.linspace(0, 1, round(1 / h) + 1)
    values = np.ones(len(times))
    for i in range(len(times) - 1):
        values[i + 1] = values[i] + h * 2 * values[i]
    np.testing.assert_allclose(values, (1 + 2 * h) ** np.arange(len(times)))
    errors.append(np.exp(2) - values[-1])
    ax.plot(times, values, color=color, marker=marker, markersize=3.5,
            linewidth=1.2, linestyle="--", label=rf"Euler: $h={h:g}$")
assert all(a > b > 0 for a, b in zip(errors, errors[1:]))
exact_value = np.exp(2 * 0.8)
euler_value = (1 + 2 * 0.2) ** 4
ax.annotate("", xy=(0.8, exact_value), xytext=(0.8, euler_value),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2,
                            shrinkA=0, shrinkB=0))
ax.text(0.82, (exact_value + euler_value) / 2, r"$\epsilon$",
        va="center", fontsize=12, bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.set(xlabel=r"Time $t$", ylabel=r"State $x(t)$", xlim=(0, 1), ylim=(0.8, 7.7))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.savefig(output)
fig.savefig("/tmp/ode-euler-preview.png", dpi=180)
plt.close(fig)
print(f"Saved {output}; errors at t=1: {errors}")
