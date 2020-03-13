"""
"""

from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from transition import next_state, inital_state
from sparsify import sparsen_profle
from plotting import plot_profile
from sojourn import sojourn_time 


def simulate_profile(n_timepoints, init_age, age_max, stepsize=0, 
                     missing=0, sparsen=True) -> np.ndarray:
    """Update the profile vector of a single female. 

    Args:
        init_age: Age at first screening.
        age_max: Age at final screening.

    Returns:
        Simulated screening history for one single female.
    """

    x = np.ones(int(n_timepoints)) * missing

    # Initial state.
    current_state = inital_state(init_age=init_age)
    
    # Track age development.
    current_age = init_age

    # Counters. 
    start_period = init_age
    end_period = 0
    num_iter = 0

    while current_age < age_max:

        # Time spent in current state.
        dt = sojourn_time(current_age, age_max, current_state)

        end_period = end_period + int(dt)
        current_age = current_age + int(dt)

        x[start_period:end_period] = current_state

        start_period = end_period
        prev_state = current_state

        # Update profile values with current state.
        current_state = next_state(age=current_age, current_state=current_state, censoring=0)

        # To avoid endless loop.
        num_iter += 1
        if num_iter > len(x):
            raise RuntimeError('Endless loop. Check config!')

    if sparsen:
        return sparsen_profle(x, init_age, age_max, stepsize=stepsize, missing=missing)

    return x


if __name__ == '__main__':
    # Demo run displaying profiles.

    n_timepoints = 321

    # Age at inital screening.
    proba_init_age = np.load('/Users/sela/phd/data/real/Pinit_screen_2Krandom.npy')

    # Age at final screening.
    proba_dropout = np.load('/Users/sela/phd/data/real/Pdropout_2Krandom.npy')

    #time = np.linspace(16, 96, n_timepoints)
    time = np.linspace(0, n_timepoints - 1, n_timepoints)

    _, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for axis in axes.ravel():

        init_age_idx = np.random.choice(range(n_timepoints), p=proba_init_age)

        start_age = int(time[init_age_idx])

        p = proba_init_age[init_age_idx:] / sum(proba_init_age[init_age_idx:])
        end_age = int(np.random.choice(time[init_age_idx:], p=p))

        # Sanity check.
        assert start_age < end_age + 1
        
        # Make synth screening history.
        x = simulate_profile(n_timepoints, start_age, end_age, sparsen=True, stepsize=12)

        # Add profile to figure.
        plot_profile(x, axis, show=False)

    plt.show()
