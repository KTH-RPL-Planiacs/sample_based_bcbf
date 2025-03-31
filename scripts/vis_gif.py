import math
from dataclasses import dataclass
from typing import List

import tyro
import yaml
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import scienceplots
plt.style.use('science')
import matplotlib
matplotlib.use('TkAgg')

from vis_utils import ColorMap, create_star

@dataclass
class Args:
    env_name: str = "colav_env"
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

    for file in args.data_path_list:
        traj_data = jnp.load(file)
        robot_state_traj = traj_data['robot_state_traj']
        obs_samples_esti_traj = traj_data['obs_samples_esti_traj']
        obs_samples_true_traj = traj_data['obs_samples_true_traj']
        obs_states_real_traj = traj_data['obs_states_real_traj']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, verticalalignment='top', color='black', animated=True)

        trajectory_line, = ax.plot([], [], label="Trajectory", color=ColorMap.robot_traj_colors[0][0], linewidth=3, alpha=0.5, zorder=3)

        history_length = 100
        max_alpha = 0.5
        alpha_schedule = [max_alpha*math.exp(-i/10) for i in range(history_length)]
        robot_circle_list = []
        robot_arrow_list = []
        for i in range(history_length):
            robot_circle = Circle((0, 0), (radius_robot), edgecolor=ColorMap.robot_traj_colors[0][1], facecolor=ColorMap.robot_traj_colors[0][1], alpha=alpha_schedule[i])
            robot_arrow = FancyArrow(0, 0, 0, 0, width=0.02, color=ColorMap.robot_body_color, alpha=alpha_schedule[i], length_includes_head=True)
            robot_circle_list.append(robot_circle)
            robot_arrow_list.append(robot_arrow)
        robot_circle_patches = PatchCollection(robot_circle_list, match_original=True, zorder=1)
        robot_arrow_patches = PatchCollection(robot_arrow_list, match_original=True, zorder=5)
        ax.add_collection(robot_circle_patches)
        ax.add_collection(robot_arrow_patches)

        n_obs, n_obs_samples = obs_samples_esti_traj.shape[1], obs_samples_esti_traj.shape[2]

        obs_scatter = ax.scatter([], [], s=10, color=ColorMap.obs_esti_dot_color, label="Obstacles", alpha=0.6, animated=True, zorder=5)
        obs_samples_circles = []
        for _ in range(n_obs * n_obs_samples):
            circle = Circle((0, 0), (radius_obs))
            obs_samples_circles.append(circle)
        obs_samples_patches = PatchCollection(obs_samples_circles, edgecolor=ColorMap.obs_esti_fp_color, facecolor=ColorMap.obs_esti_fp_color, alpha=0.5)
        ax.add_collection(obs_samples_patches)

        obs_scatter_true = ax.scatter([], [], s=10, color=ColorMap.obs_true_dot_color, alpha=0.4, animated=True, zorder=5)
        obs_samples_circles_true = []
        for _ in range(n_obs * n_obs_samples):
            circle = Circle((0, 0), (radius_obs))
            obs_samples_circles_true.append(circle)
        obs_samples_patches_true = PatchCollection(obs_samples_circles_true, edgecolor=ColorMap.obs_true_fp_color, facecolor=ColorMap.obs_true_fp_color, alpha=0.2)
        ax.add_collection(obs_samples_patches_true)
            
        real_obs_circles = []
        for _ in range(n_obs):
            circle = Circle((0, 0), (radius_obs), edgecolor=ColorMap.grey, facecolor=ColorMap.blue, alpha=0.5)
            real_obs_circles.append(circle)
        real_samples_patches = PatchCollection(real_obs_circles, edgecolor=ColorMap.grey, facecolor=ColorMap.blue, alpha=0.5,zorder=3)
        ax.add_collection(real_samples_patches)

        fov_wedge = Wedge(center=(0, 0), r=fov_radius,
                            theta1=0, theta2=0, edgecolor=ColorMap.grey, facecolor=ColorMap.fov_color, alpha=0.3, animated=True)
        ax.add_patch(fov_wedge) 

        def update(frame):
            trajectory_line.set_data(robot_state_traj[:frame + 1, 0], robot_state_traj[:frame + 1, 1])
            
            for i in range(history_length):
                if frame - i >= 0:
                    x, y, theta = robot_state_traj[frame - i, 0], robot_state_traj[frame - i, 1], robot_state_traj[frame - i, 2]
                    center_x, center_y = x + l_robot * jnp.cos(theta), y + l_robot * jnp.sin(theta)
                    robot_circle_list[i].set_center((center_x, center_y))
                    arrow_x, arrow_y = x - (radius_robot - l_robot) * jnp.cos(theta), y - (radius_robot - l_robot) * jnp.sin(theta)
                    arrow_length = 2*radius_robot
                    robot_arrow_list[i].set_data(x=arrow_x, y=arrow_y, dx=arrow_length*jnp.cos(theta), dy=arrow_length*jnp.sin(theta))
                else:
                    robot_circle_list[i].set_center((-10, 0))
                    robot_arrow_list[i].set_data(x=-10, y=0, dx=0, dy=0)
            robot_circle_patches.set_paths(robot_circle_list)
            robot_arrow_patches.set_paths(robot_arrow_list)
            if args.is_plot_robot_fov:
                x, y, theta = robot_state_traj[frame, 0], robot_state_traj[frame, 1], robot_state_traj[frame, 2]
                fov_center_x = x
                fov_center_y = y
                fov_wedge.set_center((fov_center_x, fov_center_y))
                fov_wedge.set_theta1(jnp.rad2deg(theta) - fov_angle / 2)
                fov_wedge.set_theta2(jnp.rad2deg(theta) + fov_angle / 2)
            else:
                fov_wedge.set_visible(False)
            
            obs_samples_coord = obs_samples_esti_traj[frame, ..., :2].reshape(-1, 2)
            for c, (x_obs, y_obs) in zip(obs_samples_circles, obs_samples_coord):
                c.set_center((x_obs, y_obs))
            obs_samples_patches.set_paths(obs_samples_circles)
            obs_scatter.set_offsets(jnp.concatenate([obs_samples_coord]))

            if args.is_plot_obs_samples_true:
                obs_samples_true_coord = obs_samples_true_traj[frame, ..., :2].reshape(-1, 2)
                for c, (x_obs, y_obs) in zip(obs_samples_circles_true, obs_samples_true_coord):
                    c.set_center((x_obs, y_obs))
                obs_samples_patches_true.set_paths(obs_samples_circles_true)
                obs_scatter_true.set_offsets(obs_samples_true_coord)
            else:
                obs_samples_patches_true.set_visible(False)
                obs_scatter_true.set_visible(False)

            if args.is_plot_obs_states_real:
                real_obs_coord = obs_states_real_traj[frame, ..., :2].reshape(-1, 2)
                for c, (x_obs, y_obs) in zip(real_obs_circles, real_obs_coord):
                    c.set_center((x_obs, y_obs))
                real_samples_patches.set_paths(real_obs_circles)
            else:
                real_samples_patches.set_visible(False)

            return [trajectory_line, robot_circle_patches, robot_arrow_patches, obs_scatter, obs_samples_patches, obs_scatter_true, obs_samples_patches_true, real_samples_patches, fov_wedge, frame_text]

        if args.env_name == "colav_env":
            target_region = create_star((3, 3), 0.2, color=(0/255, 75/255, 36/255))
            target_region.set_zorder(5)
            ax.add_patch(target_region)

        ax.set_xlabel(r"$p_x$ in $[\mathrm{m}]$", fontsize=28)
        ax.set_ylabel(r"$p_x$ in $[\mathrm{m}]$", fontsize=28)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.set_aspect('equal')
        ax.set_xlim(-3, 5)
        ax.set_ylim(-3, 5)

        frames_order = list(range(len(robot_state_traj)))
        downsample_factor = 10 
        frames_order = frames_order[::downsample_factor]
        
        anim = FuncAnimation(fig, update, frames=frames_order, blit=True, interval=100)
        plt.show()

if __name__ == "__main__":
    main(args=tyro.cli(Args))