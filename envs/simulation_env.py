from abc import ABC, abstractmethod
from typing import List

import jax
import jax.numpy as jnp

from envs.utils import single_integrator_dynamics, unicycle_dynamics, rk_step, euler_maruyama_step, State

class BaseEnv(ABC):
    def __init__(self, config):
        self.radius_robot = config['Env']['radius_robot']
        self.l_robot = config['Env']['l_robot']
        self.radius_obs = config['Env']['radius_obs']
        self.dt = config['Env']['dt']
        self.mean_obs_states = jnp.array(config['Env']['mean_obs_states'])
        self.n_obs = len(self.mean_obs_states)
        self.n_obs_samples = config['Env']['n_obs_samples']
        self.target_threshold = config['Env']['target_threshold']
        self.sigma_robot = jnp.array(config['Env']['sigma_robot'])
        self.sigma_obs = jnp.array(config['Env']['sigma_obs'])
        self.v_error = config['Env']['v_error']
        
    def reset(self, rng_reset):
        rng_robot, rng_obs = jax.random.split(rng_reset)
        obs_samples_esti, obs_states_real = self.get_obs_samples(rng_obs, self.n_obs_samples)
        obs_samples_true = obs_samples_esti.at[..., 2:].multiply(1 + self.v_error)
        obs_states_real = obs_states_real.at[..., 2:].multiply(1 + self.v_error)
        success = self.check_success(self.init_state)
        collision = self.check_collision(self.init_state, obs_states_real)
        init_noise = jax.random.uniform(rng_robot, shape=(3,), minval=self.init_state_low, maxval=self.init_state_high)
        init_robot_state = self.init_state + init_noise
        return State(
            robot_state=(init_robot_state),
            obs_samples_esti=obs_samples_esti,
            obs_samples_true=obs_samples_true,
            obs_states_real=obs_states_real,
            collision=collision,
            success=success,
            reward=self.get_reward(init_robot_state),
        )

    @abstractmethod
    def h_func(self, robot_state, obs_position):
        pass

    @abstractmethod
    def get_gmm_params(self, rng_reset):
        pass
        
    def get_obs_samples(self, rng_reset, n_obs_samples):
        assert len(self.mean_obs_states) == 1
        key, mode_weights, mode_stds, mode_means = self.get_gmm_params(rng_reset)
        key, subkey = jax.random.split(key)
        mode_indices = jax.random.categorical(subkey, jnp.log(mode_weights), shape=(n_obs_samples,))
        chosen_means = mode_means[mode_indices]
        chosen_stds = mode_stds[mode_indices]
        key, subkey = jax.random.split(key)
        noise_samples = jax.random.normal(subkey, (n_obs_samples, 4)) * chosen_stds[:, None]
        noise_samples = noise_samples.at[:, 2:].multiply(0.0)
        obs_samples = chosen_means +  noise_samples
        obs_samples = obs_samples[None, ...]
        key, subkey = jax.random.split(key)
        real_mode_idx = jax.random.categorical(subkey, jnp.log(mode_weights)).astype(int)
        real_mean = mode_means[real_mode_idx]
        key, subkey = jax.random.split(key)
        real_noise = jax.random.normal(subkey, (4,)) * mode_stds[real_mode_idx]
        real_noise = real_noise.at[2:].multiply(0.0)
        obs_state_real = real_mean + real_noise
        obs_state_real = obs_state_real[None, :]
        return obs_samples, obs_state_real

    def step(self, state: State, action: jnp.array, rng_step: jnp.array):
        rng_step, rng_robot = jax.random.split(rng_step)
        robot_state = euler_maruyama_step(unicycle_dynamics, state.robot_state, action, self.dt, rng_robot, self.sigma_robot)
        
        obs_samples_esti_positions = state.obs_samples_esti[..., :2]
        obs_samples_esti_velocities = state.obs_samples_esti[..., 2:] 
        rngs = jax.random.split(rng_step, num=(obs_samples_esti_positions.shape[0] * obs_samples_esti_positions.shape[1] + 1))
        rng_step, rng_obs = rngs[0], rngs[1:]
        rng_obs = rng_obs.reshape(obs_samples_esti_positions.shape[0], obs_samples_esti_positions.shape[1], rng_obs.shape[-1])
        obs_samples_esti_positions = jax.vmap(
            jax.vmap(euler_maruyama_step, in_axes=(None, 0, 0, None, 0, None)),
            in_axes=(None, 0, 0, None, 0, None)
        )(single_integrator_dynamics, obs_samples_esti_positions, obs_samples_esti_velocities, self.dt, rng_obs, self.sigma_obs)
        obs_samples_esti = jnp.concatenate([obs_samples_esti_positions, obs_samples_esti_velocities], axis=-1)
            
        obs_samples_true_positions = state.obs_samples_true[..., :2]
        obs_samples_true_velocities = state.obs_samples_true[..., 2:] 
        # Assuming the same rng_obs for both obs_samples_esti and obs_samples_true
        obs_samples_true_positions = jax.vmap(
            jax.vmap(euler_maruyama_step, in_axes=(None, 0, 0, None, 0, None)),
            in_axes=(None, 0, 0, None, 0, None)
        )(single_integrator_dynamics, obs_samples_true_positions, obs_samples_true_velocities, self.dt, rng_obs, self.sigma_obs)
        obs_samples_true = jnp.concatenate([obs_samples_true_positions, obs_samples_true_velocities], axis=-1)
        
        real_obs_positions = state.obs_states_real[:, :2]
        real_obs_velocities = state.obs_states_real[:, 2:]
        rng_obs = jax.random.split(rng_step, num=real_obs_positions.shape[0])
        real_obs_positions = jax.vmap(euler_maruyama_step,
                                 in_axes=(None, 0, 0, None, 0, None)
        )(single_integrator_dynamics, real_obs_positions, real_obs_velocities, self.dt, rng_obs, self.sigma_obs)
        obs_states_real = jnp.concatenate([real_obs_positions, real_obs_velocities], axis=-1)
        
        reward = self.get_reward(robot_state)
        collision = self.check_collision(robot_state, obs_states_real)
        success = self.check_success(robot_state)
        return state.replace(robot_state=robot_state, 
                             obs_samples_esti=obs_samples_esti, 
                             obs_samples_true=obs_samples_true,
                             obs_states_real=obs_states_real, 
                             success=success, 
                             collision=collision, 
                             reward=reward)
    
    def step_deterministic(self, state: State, action: jnp.array):
        robot_state = rk_step(unicycle_dynamics, state.robot_state, action, self.dt)
        sample_obs_positions = state.obs_samples_esti[..., :2]
        sample_obs_velocities = state.obs_samples_esti[..., 2:]
        sample_obs_positions = jax.vmap(
            jax.vmap(rk_step, in_axes=(None, 0, 0, None)),
            in_axes=(None, 0, 0, None)
        )(single_integrator_dynamics, sample_obs_positions, sample_obs_velocities, self.dt)
        reward = self.get_reward(robot_state)
        obs_samples_esti = jnp.concatenate([sample_obs_positions, sample_obs_velocities], axis=-1)
        real_obs_positions = state.obs_states_real[:, :2]
        real_obs_velocities = state.obs_states_real[:, 2:]
        real_obs_positions = jax.vmap(rk_step,
                                 in_axes=(None, 0, 0, None)
        )(single_integrator_dynamics, real_obs_positions, real_obs_velocities, self.dt)
        obs_states_real = jnp.concatenate([real_obs_positions, real_obs_velocities], axis=-1)
        collision = self.check_collision(robot_state, obs_states_real)
        success = self.check_success(robot_state)
        return state.replace(robot_state=robot_state, 
                             obs_samples_esti=obs_samples_esti, 
                             obs_states_real=obs_states_real, 
                             success=success, 
                             collision=collision, 
                             reward=reward) 
        
    def check_collision(self, robot_state, obs_states_real):
        robot_center = robot_state[:2] + self.l_robot * jnp.array([jnp.cos(robot_state[2]), jnp.sin(robot_state[2])])
        distance = jnp.linalg.norm(robot_center - obs_states_real[:, :2], axis=1)
        collision = jnp.any(distance < (self.radius_robot + self.radius_obs))
        return collision

    def check_success(self, robot_state):
        return jnp.linalg.norm(robot_state[:2] - self.target_state[:2]) < self.target_threshold

    def get_reward(self, robot_state):
        reward_target = -jnp.linalg.norm(robot_state[:2] - self.target_state[:2])
        return reward_target
    
    def save_data(self, trajectory: List[State], save_path: str, collision_frame):
        obs_samples_esti_traj = jnp.stack([state.obs_samples_esti for state in trajectory])
        obs_samples_true_traj = jnp.stack([state.obs_samples_true for state in trajectory])
        obs_states_real_traj = jnp.stack([state.obs_states_real for state in trajectory])
        robot_state_traj = jnp.stack([state.robot_state for state in trajectory])
        jnp.savez(save_path, 
                  obs_samples_esti_traj=obs_samples_esti_traj, 
                  obs_samples_true_traj=obs_samples_true_traj, 
                  obs_states_real_traj=obs_states_real_traj, 
                  robot_state_traj=robot_state_traj, 
                  collision_frame=collision_frame)
        
class ColAvEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.init_state = jnp.array([-1, -1, 0])
        self.init_state_low = jnp.array([-0.1, -0.1, jnp.pi/6])
        self.init_state_high = jnp.array([0.1, 0.1, jnp.pi/3])
        self.target_state = jnp.array([3, 3, jnp.pi/3])

    def get_gmm_params(self, rng_reset):
        key, subkey = jax.random.split(rng_reset)
        n_modes = 3
        mode_weights = jnp.array([0.7, 0.15, 0.15])
        mode_stds = jnp.array([0.05, 0.03, 0.03])
        base_noise = jax.random.normal(subkey, (n_modes, 4))
        base_noise = base_noise.at[:, :2].multiply(1.3)
        base_noise = base_noise.at[:, 2:].multiply(0.0)
        single_mean = self.mean_obs_states[0]
        mode_means = single_mean + base_noise
        mode_means = mode_means.at[0, :].set(self.mean_obs_states[0])
        return key, mode_weights, mode_stds, mode_means

    def h_func(self, robot_state, obs_position):
        h_val =  (robot_state[0] + self.l_robot * jnp.cos(robot_state[2]) - obs_position[0]) ** 2 \
                + (robot_state[1] + self.l_robot * jnp.sin(robot_state[2]) - obs_position[1]) ** 2 \
                - (self.radius_robot + self.radius_obs) ** 2
        h_val = jnp.array([h_val])
        return h_val

class TrackingEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.fov_angle = config['Env']['fov_angle']
        self.init_state = jnp.array([0, 0, 0.5*jnp.pi])
        self.init_state_low = jnp.array([-0.1, -0.1, 0])
        self.init_state_high = jnp.array([0.1, 0.1, 0])
        self.target_state = self.init_state

    def get_gmm_params(self, rng_reset):
        key, subkey = jax.random.split(rng_reset)
        n_modes = 2
        mode_weights = jnp.array([0.85, 0.15,])
        mode_stds = jnp.array([0.05, 0.03])
        mode_means = jnp.zeros((n_modes, 4))
        mode_means = mode_means.at[0, :].set(jnp.array([0.0, 5.0, 0.75, -0.75]))
        mode_means = mode_means.at[1, :].set(jnp.array([0.0, 3.0, 0.75, -0.75]))
        return key, mode_weights, mode_stds, mode_means
        
    def h_func(self, robot_state, obs_position):
        robot_cam_pos_x = robot_state[0]
        robot_cam_pos_y = robot_state[1]
        local_obs_position = obs_position - jnp.array([robot_cam_pos_x, robot_cam_pos_y])
        theta = robot_state[2]
        local_obs_position = jnp.array([jnp.cos(theta)*local_obs_position[0] + jnp.sin(theta)*local_obs_position[1],
                                       -jnp.sin(theta)*local_obs_position[0] + jnp.cos(theta)*local_obs_position[1]])
        fov_angle_rad = jnp.deg2rad(self.fov_angle)
        h_val = jnp.array([
            jnp.tan(fov_angle_rad/2) * local_obs_position[0] - self.radius_obs/jnp.cos(fov_angle_rad/2) + local_obs_position[1],
            jnp.tan(fov_angle_rad/2) * local_obs_position[0] - self.radius_obs/jnp.cos(fov_angle_rad/2) - local_obs_position[1],
        ]) # corresponds to the `n_cbf` in yml file
        return h_val