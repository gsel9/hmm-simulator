"""Top-level simulation entry point for the HMM simulator.

Provides simulate_profile() to generate a single synthetic screening
history as a sequence of clinical states over time.
"""

import numpy as np
import matplotlib.pyplot as plt

from .transition import next_state, initial_state
from .plotting import plot_profile
from .sojourn import sojourn_time


def simulate_profile(
    n_timepoints, init_age, age_max, missing=0
) -> np.ndarray:
    """Simulate the screening history of a single female.

    Args:
        n_timepoints: Total length of the output array.
        init_age: Timepoint index at first screening.
        age_max: Timepoint index at final screening.
        missing: Fill value for unobserved timepoints.

    Returns:
        Simulated screening history as an array of length n_timepoints.

    Raises:
        RuntimeError: If the simulation enters an endless loop.
    """
    x = np.ones(int(n_timepoints)) * missing

    current_state = initial_state(init_age=init_age)
    current_age = init_age

    start_period = init_age
    end_period = 0
    num_iter = 0

    while current_age < age_max:
        dt = sojourn_time(current_age, age_max, current_state)

        end_period = end_period + int(dt)
        current_age = current_age + int(dt)

        x[start_period:end_period] = current_state

        start_period = end_period

        if current_age < age_max:
            current_state = next_state(
                age=current_age, current_state=current_state, censoring=0
            )

        num_iter += 1
        if num_iter > len(x):
            raise RuntimeError('Endless loop. Check config!')

    return x


if __name__ == '__main__':
    n_timepoints = 321

    proba_init_age = np.ones(n_timepoints) / n_timepoints
    proba_dropout = np.ones(n_timepoints) / n_timepoints

    time = np.linspace(0, n_timepoints - 1, n_timepoints)

    _, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for axis in axes.ravel():
        init_age_idx = np.random.choice(range(n_timepoints), p=proba_init_age)
        start_age = int(time[init_age_idx])

        p = proba_dropout[init_age_idx:] / sum(proba_dropout[init_age_idx:])
        end_age = int(np.random.choice(time[init_age_idx:], p=p))

        assert start_age < end_age + 1

        x = simulate_profile(n_timepoints, start_age, end_age)
        plot_profile(x, axis, show=False)

    plt.show()
