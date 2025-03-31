import flax
import jax
import jax.numpy as jnp

def single_integrator_dynamics(state):
    # state: [x, y], action: [v_x, v_y]
    f = jnp.zeros_like(state)
    g = jnp.eye(2)
    return f, g

def unicycle_dynamics(state):
    # state: [x, y, theta], action: [v, w] 
    theta = state[2]
    f = jnp.zeros_like(state)
    g = jnp.array([[jnp.cos(theta), 0],
                   [jnp.sin(theta), 0],
                   [0, 1]])
    return f, g

def dyn_ode(dyn_affine, state, action):
    f, g = dyn_affine(state)
    return f + g @ action

def rk_step(dynamics, state, action, dt):
    k1 = dyn_ode(dynamics, state, action)
    k2 = dyn_ode(dynamics, state + 0.5 * dt * k1, action)
    k3 = dyn_ode(dynamics, state + 0.5 * dt * k2, action)
    k4 = dyn_ode(dynamics, state + dt * k3, action)
    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state

def euler_maruyama_step(dynamics, state, action, dt, rng_key, sigma):
    k1 = dyn_ode(dynamics, state, action)
    noise = jax.random.normal(rng_key, shape=state.shape)
    next_state = state + k1*dt + sigma*noise*jnp.sqrt(dt)
    return next_state

@flax.struct.dataclass
class State:
    robot_state: jnp.ndarray # (3,), unicycle model
    obs_samples_esti: jnp.ndarray # (n_obs, n_obs_samples, 4), single integrator model
    obs_samples_true: jnp.ndarray # (n_obs, n_obs_samples, 4), single integrator model
    obs_states_real: jnp.ndarray # (n_obs, 4)
    reward: jnp.ndarray
    collision: jax.Array
    success: jax.Array