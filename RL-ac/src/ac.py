import jax
import jax.numpy as jnp

def create_parameters(rng_key, observation):
    params = {}
    rng_key, param_key = jax.random.split(rng_key)
    params['w'] = jax.random.truncated_normal(param_key, -1, 1, (jnp.size(observation), 50))
    rng_key, param_key = jax.random.split(rng_key)
    params['b'] = jax.random.truncated_normal(param_key, -1, 1, (50,))
    rng_key, param_key = jax.random.split(rng_key)
    params['w_p'] = jax.random.truncated_normal(param_key, -1, 1, (50, 3))
    rng_key, param_key = jax.random.split(rng_key)
    params['b_p'] = jax.random.truncated_normal(param_key, -1, 1, (3,))
    rng_key, param_key = jax.random.split(rng_key)
    params['w_v'] = jax.random.truncated_normal(param_key, -1, 1, (50, 1))
    rng_key, param_key = jax.random.split(rng_key)
    params['b_v'] = jax.random.truncated_normal(param_key, -1, 1, (1,))
    return params

def network(params, observation):
    flat_observation = jnp.ravel(observation)
    h = jax.nn.relu(flat_observation.dot(params['w']) + params['b'])
    p = h.dot(params['w_p']) + params['b_p']
    v = h.dot(params['w_v']) + params['b_v']
    v = v.squeeze()
    return v, p

@jax.jit
def softmax_policy(parameters, key, obs):
    _, p = network(parameters, obs)
    return jax.random.categorical(key, p)

def policy_gradient(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t):
    def loss_fn(params, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        v_tm1, logits_tm1 = network(params, obs_tm1)
        v_t, _ = network(parameters, obs_t)
        q_tm1 = r_t + discount_t * v_t
        td_error = q_tm1 - v_tm1
        log_prob_tm1 = jax.nn.log_softmax(logits_tm1)[a_tm1]
        return log_prob_tm1 * td_error
    p_grads = jax.grad(loss_fn, argnums=0)(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t)
    return p_grads

def value_update(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t):
    v_tm1, _ = network(parameters, obs_tm1)
    _, _ = network(parameters, obs_t)
    v_t, _ = network(parameters, obs_t)
    td_target = r_t + discount_t * v_t
    td_error = td_target - v_tm1
    value_loss_fn = lambda parameters: jnp.mean(jnp.square(network(parameters, obs_tm1)[0] - td_target))
    v_grads = jax.grad(value_loss_fn)(parameters)
    return v_grads

@jax.jit
def compute_gradient(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t):
    pgrads = policy_gradient(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t)
    td_update = value_update(parameters, obs_tm1, a_tm1, r_t, discount_t, obs_t)
    return jax.tree_map(lambda pg, td: pg + td, pgrads, td_update)

alpha = 0.003
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-3

def opt_update(grads, opt_state):
    mu = opt_state['mu']
    nu = opt_state['nu']
    mu = jax.tree_map(lambda m, g: beta_1 * m + (1 - beta_1) * g, mu, grads)
    nu = jax.tree_map(lambda n, g: beta_2 * n + (1 - beta_2) * jnp.square(g), nu, grads)
    updates = jax.tree_map(lambda m, n: alpha * m / (jnp.sqrt(n) + epsilon), mu, nu)
    opt_state = {'mu': mu, 'nu': nu}
    return updates, opt_state

def opt_init(parameters):
    opt_state = {'mu': jax.tree_map(jnp.zeros_like, parameters), 'nu': jax.tree_map(jnp.ones_like, parameters)}
    return opt_state
