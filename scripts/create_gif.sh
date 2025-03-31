#!/bin/zsh

python3 scripts/vis_gif.py --env_name "tracking_env" \
                           --data_path_list "results/tracking_env_BeliefCBF_VaR_seed_42_v_error_0_N_200.npz" \
                           --is-plot-robot-fov

python3 scripts/vis_gif.py --env_name "colav_env" \
                           --data_path_list "results/colav_env_BeliefCBF_VaR_seed_42_v_error_0_N_200.npz" \
                           --no-is-plot-robot-fov