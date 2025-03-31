from abc import ABC, abstractmethod
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binom
import proxsuite

from envs.utils import unicycle_dynamics

class VanillaMPPI(ABC):
    def __init__(self, config, env):
        self.device = jax.devices(config['ShieldMPPI']['device'])[0]
        self.radius_robot = config['Env']['radius_robot']
        self.l_robot = config['Env']['l_robot']
        self.radius_obs = config['Env']['radius_obs']
        self.n_x = config['Env']['n_x']
        self.n_u = config['Env']['n_u']
        self.n_obs_samples = config['Env']['n_obs_samples']
        self.n_obs = env.n_obs
        self.mppi_horizon = int(config['ShieldMPPI']['time_horizon'] / config['Env']['dt'])
        self.actions = jnp.zeros((self.mppi_horizon, config['Env']['n_u']), device=self.device)
        self.step_env_jit = jax.jit(env.step_deterministic, device=self.device)
        self.mppi_jit = jax.jit(partial(self.mppi, self.step_env_jit, config), device=self.device)
    
    def reset(self):
        self.actions = jnp.zeros_like(self.actions, device=self.device)
    
    def mppi(self, step_env, config, state, actions, rng_action):
        n_u = config['Env']['n_u']
        n_mppi_samples = config['ShieldMPPI']['n_mppi_samples']
        temperature = config['ShieldMPPI']['temperature']
        def rollout_env(step_env, state, actions):
            def step(state, action):
                state = step_env(state, action)
                return state, (state.robot_state, state.reward)
            _, (robot_states, rewards) = jax.lax.scan(step, state, actions)
            return robot_states, rewards
        noise = jax.random.normal(rng_action, (n_mppi_samples, self.mppi_horizon, n_u))
        actions_batch = actions + noise
        robot_states_batch, rewards_batch = jax.vmap(rollout_env, in_axes=(None, None, 0))(step_env, state, actions_batch)
        rewards = rewards_batch.mean(axis=-1)
        rewards_std = rewards.std() 
        rewards_std = jnp.where(rewards_std < 1e-4, 1.0, rewards_std)        
        rew_standardized = (rewards - rewards.mean()) / rewards_std
        weights = jax.nn.softmax(rew_standardized / temperature)
        actions = jnp.einsum('n, nij->ij', weights, actions_batch)
        actions = jnp.roll(actions, shift=-1, axis=0)
        actions = actions.at[-1].set(actions[-2])
        return actions
    
    def get_action(self, state, rng_action):
        state, rng_action = jax.block_until_ready(jax.device_put((state, rng_action), self.device))
        rng_action, rng_action_mppi = jax.random.split(rng_action)
        t0 = time.perf_counter()
        self.actions = jax.block_until_ready(self.mppi_jit(state, self.actions, rng_action_mppi))
        t1 = time.perf_counter()
        timings = [(t1 - t0) * 1000, 1e-3, 1e-3, 1e-3]
        return self.actions[0], timings, state, rng_action
    
class ShieldMPPI(VanillaMPPI):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.sigma_robot = jnp.array(config['Env']['sigma_robot'])
        self.sigma_obs = jnp.array(config['Env']['sigma_obs'])
        self.h_func = env.h_func
        self.qp_solver = None
        self.qp_initialized = False
        self.compute_qp_params_jit = jax.jit(self.compute_qp_params, device=self.device)
        # self.compute_qp_params_jit = self.compute_qp_params # for debugging

    @abstractmethod
    def compute_qp_params(self, robot_state, obs_samples):
        pass
    
    def cbf_qp(self, state, ref_action):
        t0 = time.perf_counter()
        C, ub, lb = jax.block_until_ready(self.compute_qp_params_jit(state.robot_state, state.obs_samples_esti))
        t1 = time.perf_counter()
        C, ub, lb = jax.device_get((C, ub, lb,))
        C = np.array(C)
        ub = np.array(ub)
        lb = np.array(lb)
        t2 = time.perf_counter()
        H = 2 * np.eye(2)
        g = -2 * np.array(ref_action)
        H[0][0] *= 10
        g[0] *= 10
        if not self.qp_initialized:
            self.qp_solver.init(H, g, None, None, C, lb, ub) 
        else:
            self.qp_solver.update(H, g, None, None, C, lb, ub) 
        self.qp_solver.solve()
        t3 = time.perf_counter()
        calc_param_time = (t1 - t0) * 1000
        calc_transport_time = (t2 - t1) * 1000
        calc_qp_time = (t3 - t2) * 1000
        return self.qp_solver.results.x, (calc_param_time, calc_transport_time, calc_qp_time)

    def get_action(self, state, rng_action):
        ref_action, timings_mppi, state, rng_action = super().get_action(state, rng_action)
        rng_action, rng_action_cbf = jax.random.split(rng_action)
        safe_action, timings_cbf = self.cbf_qp(state, ref_action)
        timings = [timings_mppi[0], *timings_cbf]
        return safe_action, timings, state, rng_action

class BeliefCBF(ShieldMPPI):
    def __init__(self, config, env):
        super().__init__(config, env)
        n_ineq = self.n_obs * config["Env"]["n_cbf"]
        self.qp_solver = proxsuite.proxqp.dense.QP(self.n_u, 0, n_ineq)
        self.qp_initialized = False

class BeliefCBF_VaR(BeliefCBF):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.delta = config["BeliefCBF_VaR"]["delta"]
        self.tau = config["BeliefCBF_VaR"]["tau"]
        self.gamma = config["BeliefCBF_VaR"]["gamma"]
        assert ((self.n_obs_samples < jnp.log(self.delta)/jnp.log(1-self.tau)) == False).all(), "n_obs_samples is too small for BeliefCBF_VaR"
        self.k = self.compute_k(self.tau, self.n_obs_samples, self.delta)
    
    def compute_k(self, tau, N, delta):
        k_values = np.arange(1, N + 1) 
        binom_cdf = binom.cdf(k_values - 1, N, 1-tau) 
        valid_k = k_values[binom_cdf >= 1 - delta]  
        return int(valid_k[0]) if valid_k.size > 0 else N  
    
    def compute_qp_params(self, robot_state, obs_samples):
        f, g = unicycle_dynamics(robot_state)
        def compute_single_obs(robot_state, single_obs_positions, single_obs_velocities):
            ksi_vals = jax.vmap(self.h_func, in_axes=(None, 0))(robot_state, single_obs_positions)
            def compute_h_1d(ksi_vals_1d, ksi_dim, robot_state, single_obs_positions, single_obs_velocities):
                sort_indices = jnp.argsort(ksi_vals_1d, descending=True)
                active_obs_indice = sort_indices[self.k]
                h_belief = ksi_vals_1d[active_obs_indice]
                dh_dx = jax.jacfwd(self.h_func, argnums=0)(robot_state, single_obs_positions[active_obs_indice])[ksi_dim]
                d2h_dx2 = jax.hessian(self.h_func, argnums=0)(robot_state, single_obs_positions[active_obs_indice])[ksi_dim]
                dh_do = jax.jacfwd(self.h_func, argnums=1)(robot_state, single_obs_positions[active_obs_indice])[ksi_dim]
                d2h_do2 = jax.hessian(self.h_func, argnums=1)(robot_state, single_obs_positions[active_obs_indice])[ksi_dim]
                Sigma_robot = jnp.diag(self.sigma_robot)
                Sigma_obs = jnp.diag(self.sigma_obs)
                trace_term_robot = jnp.trace(Sigma_robot.T @ d2h_dx2 @ Sigma_robot)
                trace_term_obss = jnp.trace(Sigma_obs.T @ d2h_do2 @ Sigma_obs)
                trace_term = 0.5 * (trace_term_robot + jnp.sum(trace_term_obss))
                correction_term = (jnp.sum((dh_dx @ Sigma_robot) ** 2) + jnp.sum((dh_do @ Sigma_obs) ** 2)) / h_belief
                C = (dh_dx @ g).flatten()
                ub = (jnp.ones_like(h_belief) * jnp.inf)              
                lb = (-self.gamma * (h_belief)**3 + correction_term - trace_term - dh_dx @ f - jnp.sum(dh_do * single_obs_velocities[active_obs_indice]))
                return C, ub, lb
            ksi_dims = jnp.arange(ksi_vals.shape[-1])
            C, ub, lb = jax.vmap(compute_h_1d, in_axes=(1, 0, None, None, None))(ksi_vals, ksi_dims, robot_state, single_obs_positions, single_obs_velocities)            
            return C, ub, lb
        obs_positions = obs_samples[..., :2]
        obs_velocities = obs_samples[..., 2:]
        C, ub, lb = jax.vmap(compute_single_obs, in_axes=(None, 0, 0))(robot_state, obs_positions, obs_velocities)
        C = C.reshape(-1, self.n_u)
        ub = ub.flatten()
        lb = lb.flatten()
        return C, ub, lb

class BeliefCBF_CVaR(BeliefCBF):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.delta = config["BeliefCBF_CVaR"]["delta"]
        self.tau = config["BeliefCBF_CVaR"]["tau"]
        self.gamma = config["BeliefCBF_CVaR"]["gamma"]
        self.b_max = config["BeliefCBF_CVaR"]["b_max"]
        assert ((self.n_obs_samples < -0.5*jnp.log(self.delta)/(1-self.tau)**2) == False).all(), "n_obs_samples is too small for BeliefCBF_CVaR"

    def compute_qp_params(self, robot_state, obs_samples):
        f, g = unicycle_dynamics(robot_state)
        def compute_single_obs(robot_state, single_obs_positions, single_obs_velocities):
            def value_grad_hessian(robot_state, single_obs_position):
                ksi_val = self.h_func(robot_state, single_obs_position)
                dksi_dx, dksi_do = jax.jacfwd(self.h_func, argnums=(0, 1))(robot_state, single_obs_position)
                hessian_ksi = jax.hessian(self.h_func, argnums=(0, 1))(robot_state, single_obs_position)
                d2ksi_dx2, d2ksi_do2 = hessian_ksi[0][0], hessian_ksi[1][1]
                return ksi_val, (dksi_dx, dksi_do), (d2ksi_dx2, d2ksi_do2)
            ksi_vals, (dksi_dxs, dksi_dos), (d2ksi_dx2s, d2ksi_do2s) = jax.vmap(value_grad_hessian, in_axes=(None, 0))(robot_state, single_obs_positions)
            def compute_h_1d(ksi_vals_1d, dksi_dxs_1d, dksi_dos_1d, d2ksi_dx2s_1d, d2ksi_do2s_1d):
                sort_indices = jnp.argsort(-ksi_vals_1d)
                ksi_vals_1d = ksi_vals_1d[sort_indices]
                dksi_dxs_1d = dksi_dxs_1d[sort_indices]
                dksi_dos_1d = dksi_dos_1d[sort_indices]
                d2ksi_dx2s_1d = d2ksi_dx2s_1d[sort_indices]
                d2ksi_do2s_1d = d2ksi_do2s_1d[sort_indices]
                ksi_vals_1d = jnp.concatenate((ksi_vals_1d, jnp.array([self.b_max])))
                i_values = jnp.arange(1, self.n_obs_samples + 1)
                prefactor = i_values / self.n_obs_samples - jnp.sqrt(jnp.log(1 / self.delta) / (2 * self.n_obs_samples)) - (1 - self.tau)
                prefactor = (1 / self.tau) * jnp.clip(prefactor, a_min=0)
                summation = jnp.sum((ksi_vals_1d[:-1] - ksi_vals_1d[1:]) * prefactor)
                h_belief = self.b_max + summation
                dh_dksis = jnp.concatenate((prefactor[0:1], prefactor[1:] - prefactor[:-1]))
                dh_dx = jnp.sum(dh_dksis[:, None] * dksi_dxs_1d, axis=0)
                d2h_dx2 = jnp.sum(dh_dksis[:, None, None] * d2ksi_dx2s_1d, axis=0)
                dh_dos = dh_dksis[:, None] * dksi_dos_1d
                d2h_do2s = dh_dksis[:, None, None] * d2ksi_do2s_1d
                Sigma_robot = jnp.diag(self.sigma_robot)
                Sigma_obs = jnp.diag(self.sigma_obs)
                trace_term_robot = jnp.trace(Sigma_robot.T @ d2h_dx2 @ Sigma_robot)
                trace_term_obss = jax.vmap(lambda d2h_do2, Sigma_obs: jnp.trace(Sigma_obs.T @ d2h_do2 @ Sigma_obs),
                                           in_axes=(0, None))(d2h_do2s, Sigma_obs)
                trace_term = 0.5 * (trace_term_robot + jnp.sum(trace_term_obss))
                correction_term = (jnp.sum((dh_dx @ Sigma_robot) ** 2) + jnp.sum((dh_dos @ Sigma_obs) ** 2)) / h_belief
                C = (dh_dx @ g).reshape(-1, self.n_u).flatten()
                ub = (jnp.ones_like(h_belief) * jnp.inf)
                lb = (-self.gamma * (h_belief)**3 + correction_term - trace_term - dh_dx @ f - jnp.sum(dh_dos * single_obs_velocities))
                return C, ub, lb
            C, ub, lb = jax.vmap(compute_h_1d, in_axes=(1,1,1,1,1))(ksi_vals, dksi_dxs, dksi_dos, d2ksi_dx2s, d2ksi_do2s)
            return C, ub, lb
        obs_positions = obs_samples[..., :2]
        obs_velocities = obs_samples[..., 2:]
        C, ub, lb = jax.vmap(compute_single_obs, in_axes=(None, 0, 0))(robot_state, obs_positions, obs_velocities)
        C = C.reshape(-1, self.n_u)
        ub = ub.flatten()
        lb = lb.flatten()
        return C, ub, lb

class BeliefCBF_E(BeliefCBF_CVaR):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.tau = 1.0
        self.delta = config["BeliefCBF_E"]["delta"]
        self.gamma = config["BeliefCBF_E"]["gamma"]
        self.b_max = config["BeliefCBF_E"]["b_max"]
        assert ((self.n_obs_samples < -0.5*jnp.log(self.delta)) == False).all(), "n_obs_samples is too small for BeliefCBF_E"


        
           
