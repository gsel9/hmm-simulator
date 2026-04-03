# HMM Simulator

A Python implementation of a Hidden Markov Model simulator for cervical cancer screening histories.

The model generates synthetic longitudinal screening records by simulating how individuals move through clinical states over time, using age-stratified transition intensities and piecewise-exponential sojourn time distributions.

## States

| Value | Label | Meaning |
|-------|-------|---------|
| 0 | — | Censored / missing |
| 1 | N0 | Normal |
| 2 | L1 | Low-grade lesion |
| 3 | H2 | High-grade lesion |
| 4 | C3 | Carcinoma |

## Project structure

```
hmm-simulator/
├── src/
│   └── hmm_simulator/
│       ├── __init__.py
│       ├── hmm_generator.py   # top-level simulation entry point
│       ├── transition.py      # state transition logic
│       ├── sojourn.py         # sojourn time distributions
│       ├── utils.py           # parameters and helpers
│       ├── sparsify.py        # subsample dense profiles
│       └── plotting.py        # visualisation
├── tests/                     # pytest test suite
├── examples/
│   └── basic_usage.ipynb      # interactive walkthrough
├── pyproject.toml
└── LICENSE
```

## Installation

From the repository root:

```bash
pip install -e .
```

With dev dependencies (required to run tests):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from hmm_simulator.hmm_generator import simulate_profile
from hmm_simulator.utils import NUM_TIMEPOINTS

# Simulate one screening history (timepoint indices 0–60)
profile = simulate_profile(NUM_TIMEPOINTS, init_age=0, age_max=60)
print(profile)
```

See [examples/basic_usage.ipynb](examples/basic_usage.ipynb) for a full walkthrough covering population simulation, heatmaps, state distributions, and sparse screening.

## Running tests

```bash
pytest
```

## Reference

This simulator is based on the model described in:

Soper, B. C., Nygård, M., Abdulla, G., Meng, R., & Nygård, J. F. (2020).
A hidden Markov model for population-level cervical cancer screening data.
*Statistics in Medicine*, 39(25), 3569–3590.
https://doi.org/10.1002/sim.8697

```bibtex
@article{soper2020hidden,
  title={A hidden Markov model for population-level cervical cancer screening data},
  author={Soper, Braden C and Nyg{\aa}rd, Mari and Abdulla, Ghaleb and Meng, Rui and Nyg{\aa}rd, Jan F},
  journal={Statistics in Medicine},
  volume={39},
  number={25},
  pages={3569--3590},
  year={2020},
  publisher={Wiley Online Library}
}
```

## License

MIT
