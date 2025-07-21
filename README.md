# Control Barrier Functions for Sample-Based Beliefs

This repository accompanies our paper:  ["Risk-Aware Robot Control in Dynamic Environments using Belief Control Barrier Functions"](https://arxiv.org/abs/2504.04097), accepted at CDC 2025, Rio de Janeiro, Brazil.


## Why This Tool?

Real-world robots operate under **stochastic uncertainties** — caused by unmodeled dynamics, noisy sensors, and partial observability.

This tool is designed to **guarantee safety using only i.i.d. samples from belief distributions**, without relying on **parametric representations**.

### ✅ Key Features
- **Provable safety guarantees** under uncertainty
- **No parametric modeling required** — works directly with i.i.d. samples from any Bayesian state estimator
- **Real-time performance** at **kilo-Hz rates**, suitable for robotics


## Installation
Clone the repository:

```zsh
git clone https://github.com/KTH-RPL-Planiacs/sample_based_bcbf
cd sample_based_bcbf
```

Create and activate the virtual environment using [Mamba](https://mamba.readthedocs.io/en/latest/index.html).
```zsh
mamba env create -f environment.yml
mamba activate sample_based_bcbf
```


## How to Run
Run the main experiment and collect data:
```zsh
python3 main.py 
```
Visualize results with animation:
```zsh
zsh scripts/create_gif.sh
```

## BibTeX
If you find this work useful, please consider citing:
```
@article{han2025risk,
  title={Risk-Aware Robot Control in Dynamic Environments Using Belief Control Barrier Functions},
  author={Han, Shaohang and Vahs, Matti and Tumova, Jana},
  journal={arXiv preprint arXiv:2504.04097},
  year={2025}
}
```
