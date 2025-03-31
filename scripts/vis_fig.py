import math
from dataclasses import dataclass
from typing import List

import tyro
import yaml
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import scienceplots
plt.style.use('science')
import matplotlib
matplotlib.use('TkAgg')

from vis_utils import ColorMap, create_star

@dataclass
class Args:
    env_name: str = "colav_env"
    is_plot_robot_state: bool = False
    is_plot_robot_fov: bool = False
    is_plot_obs_samples_true: bool = False
    is_plot_obs_states_real: bool = False
    data_path_list: List[str] = ()
    key_frames: List[int] = ()

def main(args: Args):
    for data_path in args.data_path_list:
        if (args.env_name in data_path) == False:
            raise ValueError("Data path does not match env_name")
    
    with open(f"configs/{args.env_name}.yml", "r") as file:
        config = yaml.safe_load(file)
    radius_robot = config['Env']['radius_robot']
    l_robot = config['Env']['l_robot']
    radius_obs = config['Env']['radius_obs']
    fov_radius = config['Env']['fov_radius']
    fov_angle = config['Env']['fov_angle']

    # plot obstacles, which should be the same for all data files
    fig, ax = plt.subplots(figsize=(8, 8))
    traj_data = jnp.load(args.data_path_list[0])
    obs_samples_esti_traj = traj_data['obs_samples_esti_traj']
    obs_samples_true_traj = traj_data['obs_samples_true_traj']
    obs_states_real_traj = traj_data['obs_states_real_traj']
    for i in args.key_frames:
        obs_samples_esti_coord = obs_samples_esti_traj[i, ..., :2].reshape(-1, 2)
        obs_samples_esti_circles = []
        for (x_obs, y_obs) in obs_samples_esti_coord:
            circle = Circle((x_obs, y_obs), (radius_obs))
            obs_samples_esti_circles.append(circle)
        obs_samples_esti_patches = PatchCollection(obs_samples_esti_circles, edgecolor=ColorMap.obs_esti_fp_color, facecolor=ColorMap.obs_esti_fp_color, alpha=0.5)
        ax.add_collection(obs_samples_esti_patches)
        ax.scatter(obs_samples_esti_coord[:, 0], obs_samples_esti_coord[:, 1], s=10, color=ColorMap.obs_esti_dot_color, alpha=0.6, animated=False, zorder=5)
        
        if args.is_plot_obs_samples_true:
            obs_samples_true_coord = obs_samples_true_traj[i, ..., :2].reshape(-1, 2)
            obs_samples_true_circles = []
            for (x_obs, y_obs) in obs_samples_true_coord:
                circle = Circle((x_obs, y_obs), (radius_obs))
                obs_samples_true_circles.append(circle)
            obs_samples_true_patches = PatchCollection(obs_samples_true_circles, edgecolor=ColorMap.obs_true_fp_color, facecolor=ColorMap.obs_true_fp_color, alpha=0.2)
            ax.add_collection(obs_samples_true_patches)
            ax.scatter(obs_samples_true_coord[:, 0], obs_samples_true_coord[:, 1], s=10, color=ColorMap.obs_true_dot_color, alpha=0.4, animated=False, zorder=5)

        if args.is_plot_obs_states_real:
            real_obs_coord = obs_states_real_traj[i, ..., :2].reshape(-1, 2)
            real_obs_circle = Circle((real_obs_coord[0, 0], real_obs_coord[0, 1]), (radius_obs), edgecolor=ColorMap.grey, facecolor=ColorMap.blue, alpha=0.5)
            ax.add_patch(real_obs_circle)

    # plot robot trajectory
    for num_file, file in enumerate(args.data_path_list):
        method_name = None
        if "BeliefCBF_VaR" in file:
            method_name = r"$\underline{\mathrm{VaR}_{0.1}}$"
        elif "BeliefCBF_CVaR" in file:
            method_name = r"$\underline{\mathrm{CVaR}_{0.1}}$"
        elif "BeliefCBF_E" in file:
            method_name = r"$\underline{\mathrm{E}}$"
        else:
            raise ValueError("Unknown method")
        
        traj_data = jnp.load(file)
        robot_state_traj = traj_data['robot_state_traj']
        robot_traj_color, robot_fp_color = ColorMap.robot_traj_colors[num_file]

        if args.is_plot_robot_state:
            for i in args.key_frames:
                x, y, theta = robot_state_traj[i, 0], robot_state_traj[i, 1], robot_state_traj[i, 2]    
                ax.scatter(x, y, s=70, edgecolor=ColorMap.grey, facecolor=ColorMap.robot_body_color,  zorder=5)
                # center_x, center_y = x + l_robot * jnp.cos(theta), y + l_robot * jnp.sin(theta)
                # ax.add_patch(Circle((center_x, center_y), (radius_robot), edgecolor=ColorMap.grey, facecolor=ColorMap.robot_body_color, alpha=0.5, zorder=1))
                if args.is_plot_robot_fov:
                    fov_center_x, fov_center_y = x, y
                    fov_wedge = Wedge(center=(fov_center_x, fov_center_y), r=fov_radius,
                                      theta1=jnp.rad2deg(theta) - fov_angle / 2, theta2=jnp.rad2deg(theta) + fov_angle / 2,
                                      edgecolor="black", facecolor=ColorMap.fov_color, alpha=0.3, animated=False,zorder=5)
                    ax.add_patch(fov_wedge)
        
        downsample_factor = 5
        robot_state_traj = robot_state_traj[::downsample_factor]
        ax.plot(robot_state_traj[:, 0], robot_state_traj[:, 1], color=robot_traj_color, linewidth=3, alpha=0.5, zorder=3, label=method_name)
        max_alpha = 0.5
        alpha_schedule = [max_alpha*math.exp(-i/30) for i in range(len(robot_state_traj))]
        alpha_schedule.reverse()
        robot_circle_list = []
        for i in range(len(robot_state_traj)):
            x, y, theta = robot_state_traj[i, 0], robot_state_traj[i, 1], robot_state_traj[i, 2]
            center_x, center_y = x + l_robot * jnp.cos(theta), y + l_robot * jnp.sin(theta)
            robot_circle = Circle((center_x, center_y), (radius_robot), edgecolor=robot_fp_color, facecolor=robot_fp_color, alpha=alpha_schedule[i], zorder=1)
            robot_circle_list.append(robot_circle)
        robot_circle_patches = PatchCollection(robot_circle_list, match_original=True, zorder=1)
        ax.add_collection(robot_circle_patches)

    if args.env_name == "colav_env":
        target_region = create_star((3, 3), 0.2, color=(0/255, 75/255, 36/255))
        target_region.set_zorder(5)
        ax.add_patch(target_region)

    ax.set_xlabel(r"$p_x$ in $[\mathrm{m}]$", fontsize=32)
    ax.set_ylabel(r"$p_y$ in $[\mathrm{m}]$", fontsize=32)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 7)
    ax.set_ylim(-2.5, 6.5)
    ax.legend(facecolor='white', edgecolor="black", fontsize=28, frameon=True, loc='upper left')
    plt.show()
    
if __name__ == "__main__":
    main(args=tyro.cli(Args))

    