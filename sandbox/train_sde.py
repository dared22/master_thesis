import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


def train_nsde(
    training_paths,
    training_times,
    validation_paths,
    validation_times,
    *,
    model_factory,
    loss_function,
    learning_rate=1e-3,
    batch_size=2048,
    training_steps=1500,
    validation_every=100,
    validation_size=4096,
    seed=100,
    validation_seed=102,
    verbose=True,
):
    """Train a fresh NSDE and return its best validation checkpoint and history.

    Paths have shape (n_paths, n_times). Pass actual selected timestamps.
    Validation paths must be independent of training paths. Keep validation
    data and validation_seed fixed across dataset sizes for comparable scores.
    The loss is the notebook's increment NLL with model-independent terms
    omitted. Training always runs for training_steps optimizer updates.
    """
    def prepare_increments(paths, times):
        """Flatten paths into starting positions, increments, and time steps."""
        paths, times = np.asarray(paths), np.asarray(times)
        starts = paths[:, :-1]
        increments = np.diff(paths, axis=1)
        time_steps = np.broadcast_to(np.diff(times), starts.shape)
        return tuple(
            jnp.asarray(values.reshape(-1))
            for values in (starts, increments, time_steps)
        )

    data = prepare_increments(training_paths, training_times)
    validation_data = prepare_increments(validation_paths, validation_times)
    model_key, batch_key = jr.split(jr.PRNGKey(seed))
    model = model_factory(model_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, dx, dt):
        loss, grads = eqx.filter_value_and_grad(loss_function)(
            model, x, dx, dt
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    evaluate = eqx.filter_jit(loss_function)
    
    validation_indices = jr.choice(
        jr.PRNGKey(validation_seed), validation_data[0].size,
        shape=(min(validation_size, validation_data[0].size),), replace=False,
    )
    validation_batch = tuple(values[validation_indices] for values in validation_data)
    best_loss = float(evaluate(model, *validation_batch))
    best_model, best_step = model, 0
    loss_history = []
    validation_steps = [0]
    validation_history = [best_loss]

    for step in range(1, training_steps + 1):
        batch_key, sample_key = jr.split(batch_key)
        indices = jr.randint(sample_key, (batch_size,), 0, data[0].size)
        batch = tuple(values[indices] for values in data)
        model, opt_state, loss = train_step(model, opt_state, *batch)
        batch_loss = float(loss)
        loss_history.append(batch_loss)

        if step % validation_every != 0 and step != training_steps:
            continue

        validation_loss = float(evaluate(model, *validation_batch))
        validation_steps.append(step)
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_model, best_loss, best_step = model, validation_loss, step

        if verbose:
            print(
                f"Step {step}: training = {batch_loss:.4f}, "
                f"validation = {validation_loss:.4f}"
            )

    return {
        "model": best_model,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "last_validation_loss": validation_history[-1],
        "loss_history": np.asarray(loss_history),
        "validation_steps": np.asarray(validation_steps),
        "validation_history": np.asarray(validation_history),
        "steps_completed": len(loss_history),
        "n_trajectories": int(np.shape(training_paths)[0]),
        "n_observations": int(len(training_times)),
        "n_increments": int(data[0].size),
        "seed": seed,
    }
