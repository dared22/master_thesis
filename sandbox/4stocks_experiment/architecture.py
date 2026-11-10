"""Equinox SDE generator and deterministic CDE critic for 20-session windows."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def solve_sde(
    drift: object, diffusion: object, y0: jax.Array, brownian: object,
    dt: float = 0.25,
) -> jax.Array:
    """Solve a Stratonovich SDE on the session clock, saving observations only."""
    terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, brownian))
    solution = dfx.diffeqsolve(
        terms, dfx.Heun(), t0=0.0, t1=19.0, dt0=dt, y0=y0,
        saveat=dfx.SaveAt(ts=jnp.arange(20, dtype=jnp.float32)),
        adjoint=dfx.DirectAdjoint(), max_steps=19 * round(1 / dt),
    )
    return eqx.error_if(solution.ys, ~jnp.isfinite(solution.ys).all(),
                        "Nonfinite generator solver output")


class Generator(eqx.Module):
    """Shared latent SDE with learned random initial state and linear readout."""

    initial: eqx.nn.MLP
    drift: eqx.nn.MLP
    diffusion: eqx.nn.MLP
    readout: eqx.nn.Linear
    initial_noise_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    brownian_size: int = eqx.field(static=True)

    def __init__(
        self, hidden_size: int, initial_noise_size: int, brownian_size: int,
        mlp_width: int, mlp_depth: int, *, key: jax.Array,
    ) -> None:
        keys = jr.split(key, 4)
        self.initial_noise_size = initial_noise_size
        self.hidden_size = hidden_size
        self.brownian_size = brownian_size
        self.initial = eqx.nn.MLP(initial_noise_size, hidden_size, mlp_width,
                                  mlp_depth, activation=jnp.tanh, key=keys[0])
        self.drift = eqx.nn.MLP(1 + hidden_size, hidden_size, mlp_width,
                                mlp_depth, activation=jnp.tanh,
                                final_activation=jnp.tanh, key=keys[1])
        self.diffusion = eqx.nn.MLP(
            1 + hidden_size, hidden_size * brownian_size, mlp_width, mlp_depth,
            activation=jnp.tanh, final_activation=jnp.tanh, key=keys[2],
        )
        self.readout = eqx.nn.Linear(hidden_size, 4, key=keys[3])

    def __call__(
        self, key: jax.Array, dt: float = 0.25, brownian_tol: float = 0.0625,
    ) -> jax.Array:
        """Generate one (20, 4) window; time normalization affects features only."""
        initial_key, brownian_key = jr.split(key)
        z0 = self.initial(jr.normal(initial_key, (self.initial_noise_size,)))
        brownian = dfx.VirtualBrownianTree(
            0.0, 19.0, tol=brownian_tol, shape=(self.brownian_size,),
            key=brownian_key,
        )

        def drift(t: jax.Array, z: jax.Array, args: object) -> jax.Array:
            return self.drift(jnp.concatenate((jnp.atleast_1d(t / 19), z)))

        def diffusion(t: jax.Array, z: jax.Array, args: object) -> jax.Array:
            return self.diffusion(
                jnp.concatenate((jnp.atleast_1d(t / 19), z))
            ).reshape(self.hidden_size, self.brownian_size)

        states = solve_sde(drift, diffusion, z0, brownian, dt)
        values = jax.vmap(self.readout)(states)
        return eqx.error_if(values, ~jnp.isfinite(values).all(),
                            "Nonfinite generator readout")


class Critic(eqx.Module):
    """Neural CDE scoring a complete sampled return window."""

    initial: eqx.nn.MLP
    vector_field: eqx.nn.MLP
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self, hidden_size: int, mlp_width: int, mlp_depth: int, *, key: jax.Array,
    ) -> None:
        keys = jr.split(key, 3)
        self.hidden_size = hidden_size
        self.initial = eqx.nn.MLP(4, hidden_size, mlp_width, mlp_depth,
                                  activation=jnp.tanh, key=keys[0])
        self.vector_field = eqx.nn.MLP(
            hidden_size, hidden_size * 5, mlp_width, mlp_depth,
            activation=jnp.tanh, final_activation=jnp.tanh, key=keys[1],
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=keys[2])

    def __call__(self, values: jax.Array, dt: float = 0.25) -> jax.Array:
        """Return an unrestricted scalar, using the same interpolation for all data."""
        times = jnp.arange(20, dtype=jnp.float32)
        control = dfx.LinearInterpolation(
            ts=times, ys=jnp.concatenate((times[:, None] / 19, values), axis=1),
        )

        def field(t: jax.Array, h: jax.Array, args: object) -> jax.Array:
            return self.vector_field(h).reshape(self.hidden_size, 5)

        solution = dfx.diffeqsolve(
            dfx.ControlTerm(field, control), dfx.Heun(),
            t0=0.0, t1=19.0, dt0=dt, y0=self.initial(values[0]),
            saveat=dfx.SaveAt(t1=True), adjoint=dfx.DirectAdjoint(),
            max_steps=19 * round(1 / dt),
        )
        hidden = eqx.error_if(solution.ys[0], ~jnp.isfinite(solution.ys).all(),
                              "Nonfinite critic solver output")
        score = self.readout(hidden)[0]
        return eqx.error_if(score, ~jnp.isfinite(score), "Nonfinite critic score")
