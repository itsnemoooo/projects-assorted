import jax
import jax.numpy as jnp
import numpy as np
from bsuite.environments import catch
from ac import create_parameters, softmax_policy, compute_gradient, opt_init, opt_update, apply_updates
from utils import plot_learning_curve

# Experiment configs.
train_episodes = 2500
discount_factor = .99

# Create environment.
env = catch.Catch(seed=42)

# Build and initialize network.
rng = jax.random.PRNGKey(44)
rng, init_rng = jax.random.split(rng)
sample_input = env.observation_spec().generate_value()
parameters = create_parameters(init_rng, sample_input)

# Initialize optimizer state.
opt_state = opt_init(parameters)

# Jit.
opt_update = jax.jit(opt_update)
apply_updates = jax.jit(apply_updates)

print(f"Training agent for {train_episodes} episodes...")
all_episode_returns = []

for _ in range(train_episodes):
    episode_return = 0.
    timestep = env.reset()
    obs_tm1 = timestep.observation

    # Sample initial action.
    rng, policy_rng = jax.random.split(rng)
    a_tm1 = softmax_policy(parameters, policy_rng, obs_tm1)

    while not timestep.last():
        # Step environment.
        new_timestep = env.step(int(a_tm1))

        # Sample action from agent policy.
        rng, policy_rng = jax.random.split(rng)
        a_t = softmax_policy(parameters, policy_rng, new_timestep.observation)

        # Update params.
        r_t = new_timestep.reward
        discount_t = discount_factor * new_timestep.discount

        dJ_dtheta = compute_gradient(
            parameters, obs_tm1, a_tm1, r_t, discount_t,
            new_timestep.observation)
        updates, opt_state = opt_update(dJ_dtheta, opt_state)
        parameters = apply_updates(parameters, updates)

        # Within episode book-keeping.
        episode_return += new_timestep.reward
        timestep = new_timestep
        obs_tm1 = new_timestep.observation
        a_tm1 = a_t

    # Experiment results tracking.
    all_episode_returns.append(episode_return)

# Plot learning curve.
plot_learning_curve(all_episode_returns)
