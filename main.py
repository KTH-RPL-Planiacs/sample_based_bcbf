import time
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import tyro
import logging
import yaml

import policies
import envs

jax.config.update("jax_platform_name", "cpu") # simulate in cpu
jax.config.update("jax_enable_x64", True) # use float64

@dataclass
class Args:
    seeds: List[int] = (42,) 
    sim_time: float = 10
    n_episodes: int = 1
    v_error: int = 0 # 0, 20 # in [%]
    all_env_names: List[str] = ("tracking_env", "colav_env",) # "tracking_env", "colav_env"
    all_n_obs_samples: List[int] = (200, ) # 200, 500, 1000, 5000
    all_method_names: List[str] = ("BeliefCBF_VaR",) # "BeliefCBF_VaR", "BeliefCBF_CVaR", "BeliefCBF_E"
    save_data: bool = True
    logging: bool = False

def main(args: Args):    
    if args.logging:
        log_file_path = f"results/mylog.log"
        print(f"Logging to {log_file_path}")
        logging.basicConfig(
            filename=log_file_path,    # Path to your log file
            filemode='a',              # 'w' = overwrite each run; use 'a' to append
            level=logging.INFO,        # Set the logging level
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        logging.info("****************************************")
        logging.info(f"Starting experiment with hyperparameters: {args}")
        logging.info("****************************************")

    for env_name in args.all_env_names:
        for n_obs_samples in args.all_n_obs_samples:
            with open(f"configs/{env_name}.yml", "r") as file:
                config = yaml.safe_load(file)
            n_sim_steps = int(args.sim_time / config['Env']['dt'])
            config['Env']['n_obs_samples'] = n_obs_samples
            config['Env']['v_error'] = args.v_error * 0.01
            env = envs.get_env(env_name, config) 
            step_env = jax.jit(env.step)
            reset_env = jax.jit(env.reset)
            rng_dummy = jax.random.PRNGKey(seed=-1)
            state = jax.block_until_ready(reset_env(rng_dummy)) # jitting reset_env
            dummy_action = jnp.zeros((config['Env']['n_u'],))
            state = jax.block_until_ready(step_env(state, dummy_action, rng_dummy)) # jitting step_env
            for method_name in args.all_method_names:
                control_policy = getattr(policies, method_name)(config, env)
                control_policy.get_action(state, rng_dummy) # jitting get_action
                n_collision = 0
                n_success = 0
                max_mppi_time = []
                avg_mppi_time = []
                max_cbf_time = []
                avg_cbf_time = []    
                avg_param_time = []
                avg_transport_time = []
                avg_qp_time = []
                collision_cases = set()
                timeout_cases = set()
                assert args.n_episodes == 1 if len(args.seeds) > 1 else True
                for seed in args.seeds:
                    for i in range(args.n_episodes):
                        current_seed = seed + i
                        control_policy.reset()
                        rng = jax.random.PRNGKey(seed=current_seed)
                        rng, rng_reset = jax.random.split(rng)
                        state = jax.block_until_ready(reset_env(rng_reset)) # randomized initial obstacle state
                        trajectory = [state]        
                        max_mppi_time_per_eps = 0
                        total_mppi_time_per_eps = 0
                        max_cbf_time_per_eps = 0
                        total_cbf_time_per_eps = 0
                        total_qp_time_per_eps = 0
                        total_param_time_per_eps = 0
                        total_transport_time_per_eps = 0
                        collision_flag = False
                        success_flag = False
                        collision_frame = None
                        for t in range(n_sim_steps):
                            action = jnp.zeros((config['Env']['n_u'],))
                            rng, rng_action = jax.random.split(rng)
                            action, timings, _, _ = control_policy.get_action(state, rng_action)
                            (mppi_time, calc_param_time, calc_transport_time, calc_qp_time) = timings
                            max_mppi_time_per_eps = max(max_mppi_time_per_eps, mppi_time)
                            total_mppi_time_per_eps += mppi_time
                            cbf_time = calc_param_time + calc_transport_time + calc_qp_time
                            max_cbf_time_per_eps = max(max_cbf_time_per_eps, cbf_time)
                            total_cbf_time_per_eps += cbf_time
                            total_param_time_per_eps += (calc_param_time / cbf_time)*100
                            total_transport_time_per_eps += (calc_transport_time / cbf_time)*100
                            total_qp_time_per_eps += (calc_qp_time / cbf_time)*100
                            rng, rng_step = jax.random.split(rng)
                            state = jax.block_until_ready(step_env(state, action, rng_step))
                            trajectory.append(state)
                            if state.collision:
                                collision_frame = t if collision_frame is None else collision_frame
                                collision_flag = True
                                success_flag = False
                                collision_cases.add(current_seed)
                            if collision_flag == False and (t == n_sim_steps-1) and state.success:
                                success_flag = True
                        if collision_flag == False and success_flag == False:
                            timeout_cases.add(current_seed)
                        n_collision += collision_flag
                        n_success += success_flag
                        max_mppi_time.append(max_mppi_time_per_eps)
                        avg_mppi_time.append(total_mppi_time_per_eps/ n_sim_steps)
                        max_cbf_time.append(max_cbf_time_per_eps)
                        avg_cbf_time.append(total_cbf_time_per_eps / n_sim_steps)
                        avg_param_time.append(total_param_time_per_eps / n_sim_steps)
                        avg_transport_time.append(total_transport_time_per_eps / n_sim_steps)
                        avg_qp_time.append(total_qp_time_per_eps / n_sim_steps)
                        save_path = f"results/{env_name}_{method_name}_seed_{current_seed}_v_error_{args.v_error}_N_{n_obs_samples}.npz"
                        env.save_data(trajectory, save_path, collision_frame) if args.save_data else None
                max_mppi_time = jnp.array(max_mppi_time)
                avg_mppi_time = jnp.array(avg_mppi_time)
                max_cbf_time = jnp.array(max_cbf_time)
                avg_cbf_time = jnp.array(avg_cbf_time)
                avg_param_time = jnp.array(avg_param_time)
                avg_transport_time = jnp.array(avg_transport_time)
                avg_qp_time = jnp.array(avg_qp_time)
                print("----------------------------------------")
                print(f"Running {method_name} on {env_name} with {n_obs_samples} obstacle samples")
                print("Success: %d / %d" %(n_success, args.n_episodes))
                print("Collision: %d / %d" %(n_collision, args.n_episodes))
                print("Timeout: %d / %d" %(args.n_episodes*len(args.seeds) - n_success - n_collision, args.n_episodes))
                print("Collision cases: ", ", ".join(map(str, sorted(collision_cases))))
                print("Timeout cases: ", ", ".join(map(str, sorted(timeout_cases))))
                print("Max MPPI time: %.3f ± %.3f ms" %(max_mppi_time.mean(), max_mppi_time.std()))
                print("Average MPPI time: %.3f ± %.3f ms" %(avg_mppi_time.mean(), avg_mppi_time.std()))
                print("Max CBF time: %.3f ± %.3f ms" %(max_cbf_time.mean(), max_cbf_time.std()))
                print("Average CBF time: %.3f ± %.3f ms" %(avg_cbf_time.mean(), avg_cbf_time.std()))
                print("Parameters %.1f ± %.1f %%" %(avg_param_time.mean(), avg_param_time.std()))
                print("Transport %.1f ± %.1f %%" %(avg_transport_time.mean(), avg_transport_time.std()))
                print("QP %.1f ± %.1f %%" %(avg_qp_time.mean(), avg_qp_time.std()))
                
                if args.logging:
                    logging.info("----------------------------------------")
                    logging.info(f"Running {method_name} on {env_name} with {n_obs_samples} obstacle samples")
                    logging.info("Success: %d / %d", n_success, args.n_episodes)
                    logging.info("Collision: %d / %d", n_collision, args.n_episodes)
                    logging.info("Collision cases: " + ", ".join(map(str, sorted(collision_cases))))
                    logging.info("Timeout cases: " + ", ".join(map(str, sorted(timeout_cases))))
                    logging.info("Max MPPI time: %.3f ± %.3f ms", max_mppi_time.mean(), max_mppi_time.std())
                    logging.info("Average MPPI time: %.3f ± %.3f ms", avg_mppi_time.mean(), avg_mppi_time.std())
                    logging.info("Max CBF time: %.3f ± %.3f ms", max_cbf_time.mean(), max_cbf_time.std())
                    logging.info("Average CBF time: %.3f ± %.3f ms", avg_cbf_time.mean(), avg_cbf_time.std())
                    logging.info("Parameters %.1f ± %.1f %%", avg_param_time.mean(), avg_param_time.std())
                    logging.info("Transport %.1f ± %.1f %%", avg_transport_time.mean(), avg_transport_time.std())
                    logging.info("QP %.1f ± %.1f %%", avg_qp_time.mean(), avg_qp_time.std())

if __name__ == "__main__":
    main(args=tyro.cli(Args))