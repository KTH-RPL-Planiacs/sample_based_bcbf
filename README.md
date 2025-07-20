# Control Barrier Functions for Sample-Based Beliefs

This repository accompanies the paper ["Risk-Aware Robot Control in Dynamic Environments using Belief Control Barrier Functions"](https://arxiv.org/abs/2504.04097), accepted at CDC 2025, Rio de Janeiro, Brazil.

## Why invent this tool?

Real-world robots face *stochastic uncertainties* due to factors such as unmodeled dynamics, noisy sensor measurements, and partial observability.

We aim to **guarantee safety using only i.i.d. samples from belief distributions**, *without relying on parametric representations*.

Our safety filter can run at kilo-Hz rates, ensuring **real-time performance**.

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

## Run
To run the main experiment and collect data:
```zsh
python3 main.py 
```
To visualize by animation:

```zsh
zsh scripts/create_gif.sh
```

## BibTeX
```
@article{han2025risk,
  title={Risk-Aware Robot Control in Dynamic Environments Using Belief Control Barrier Functions},
  author={Han, Shaohang and Vahs, Matti and Tumova, Jana},
  journal={arXiv preprint arXiv:2504.04097},
  year={2025}
}
```
