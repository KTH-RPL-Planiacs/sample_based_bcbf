from typing import Tuple
from dataclasses import dataclass

import jax.numpy as jnp
from matplotlib.patches import Polygon

@dataclass
class ColorMap:
    black: Tuple[float, float, float] = (0/255, 0/255, 0/255)
    red: Tuple[float, float, float] = (249/255, 238/255, 238/255)
    grey: Tuple[float, float, float] = (66/255, 65/255, 65/255)
    blue: Tuple[float, float, float] = (26/255, 51/255, 229/255)    
    
    robot_body_color: Tuple[float, float, float] = (245/255, 231/255, 113/255)
    robot_traj_colors = [
        [   # Blue
            (0/255, 40/255, 170/255),     # dot: darker
            (80/255, 160/255, 255/255)    # footprint: lighter
        ],    
        [   # Orange
            (200/255, 60/255, 0/255),     
            (255/255, 160/255, 90/255)     
        ],
        [   # Green 
            (12/255, 92/255, 31/255),
            (100/255, 215/255, 122/255)
        ],
        [   # Purple
            (130/255, 0/255, 100/255),    
            (200/255, 130/255, 255/255)   
        ],
    ]
    
    fov_color: Tuple[float, float, float] = (249/255, 238/255, 238/255)
    
    obs_esti_fp_color: Tuple[float, float, float] = (255/255, 127/255, 102/255)
    obs_esti_dot_color: Tuple[float, float, float] = (191/255, 75/255, 50/255)
    
    obs_true_fp_color: Tuple[float, float, float] =   (80/255, 130/255, 180/255)
    obs_true_dot_color: Tuple[float, float, float] = (40/255, 80/255, 120/255)
    

def create_star(center, size, color):
    x, y = center
    num_vertices = 5  # 5-pointed star
    theta = jnp.linspace(0, 2 * jnp.pi, 2 * num_vertices, endpoint=False)
    theta += jnp.pi / 2 
    r = jnp.array([size if i % 2 == 0 else size * 0.4 for i in range(2 * num_vertices)])
    x_points = x + r * jnp.cos(theta)
    y_points = y + r * jnp.sin(theta)
    points = jnp.stack([x_points, y_points], axis=1).block_until_ready()
    return Polygon(
        xy=points,
        closed=True,
        color=color,
        linewidth=0.5
    )