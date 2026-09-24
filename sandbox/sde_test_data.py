# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import matplotlib.pyplot as plt


# %%
# Modellparameter

A = 1.0
k = 1.0

sigma = 1.1

# Anzahl der Trajektorien
n_trajectories = 50


# %%
# Potential

def V(x):
    return A * (1.0 - jnp.cos(k * x))


# Drift = -dV/dx

def drift(t, x, args):
    return -A * k * jnp.sin(k * x)


# konstante Diffusion

def diffusion(t, x, args):
    return sigma


# %%
# Simulationsparameter

t0 = 0.0
t1 = 500.0
dt = 0.01

ts = jnp.linspace(
    t0,
    t1,
    15000,
)

x0 = 0.0


# %%
# Funktion für EINE Trajektorie

def simulate_one(key):

    brownian = dfx.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=1e-3,
        shape=(),
        key=key,
    )

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(
            diffusion,
            brownian,
        ),
    )

    solution = dfx.diffeqsolve(
        terms,
        dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=x0,
        saveat=dfx.SaveAt(ts=ts),
        max_steps=100_000,
    )

    return solution.ys


# %%
# n unabhängige Zufallsschlüssel erzeugen

key = jr.PRNGKey(42)

keys = jr.split(
    key,
    n_trajectories,
)


# %%
# Alle Trajektorien simulieren

trajectories = jax.vmap(
    simulate_one
)(keys)


print(trajectories.shape)

# Erwartet:
# (n_trajectories, len(ts))


# %%
# Plot

plt.figure(figsize=(12, 5))


# Alle Trajektorien
for trajectory in trajectories[:]:

    plt.plot(
        ts,
        trajectory,
        color="black",
        linewidth=0.5,
        alpha=0.5,
    )

plt.xlabel("t")
plt.ylabel("X(t)")

# plt.title(
#     f"{n_trajectories} SDE-Trajektorien im periodischen Potential"
# )

plt.grid(alpha=0.2)

plt.show()



# %%
