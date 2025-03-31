#!/bin/zsh

python3 scripts/vis_fig.py --env_name "tracking_env" \
                           --data_path_list "results/tracking_env_BeliefCBF_VaR_seed_42_v_error_0_N_200.npz" \
                           --key_frames 50 300 600 \
                           --is-plot-robot-state \
                           --is-plot-robot-fov

python3 scripts/vis_fig.py --env_name "colav_env" \
                           --data_path_list "results/colav_env_BeliefCBF_VaR_seed_42_v_error_0_N_200.npz" \
                           --key_frames 100 \
                           --no-is-plot-robot-state \
                           --no-is-plot-robot-fov

 