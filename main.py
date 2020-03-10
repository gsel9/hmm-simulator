"""
"""

from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from transition import next_state, inital_state
from sojourn import sojourn_time 


def plot_profile(x, axis, title=None, show=True, path_to_fig=None):

	if title is not None:
		plt.title(title, fontsize=22)
		
	axis.plot(x, '-o')

	axis.set_yticks(range(0, 5))
	axis.set_yticklabels(['D4', 'N0', 'L1', 'H2', 'C3'], fontsize=18)
	axis.set_ylabel('State', fontsize=20)

	axis.set_xticks(np.linspace(0, 80, 6, dtype=int))
	axis.set_xticklabels(np.linspace(16, 96, 6, dtype=int), fontsize=18)
	axis.set_xlabel('Years', fontsize=20)

	plt.tight_layout()

	if show:
		plt.show()

	if path_to_fig is not None:
		plt.savefig(path_to_fig)


def simulate_profile(init_age, age_max) -> np.ndarray:
    """Update the profile vector of a single female. 

    Args:
    	init_age: Age at first screening.
    	age_max: Age at final screening.

    Returns:
    	Simulated screening history for one single female.
    """

    x = np.ones(int(age_max - init_age)) * -1

    # Initial state.
    current_state = inital_state(init_age=init_age)
    
    # Track age development.
    current_age = init_age

    # Counters. 
    start_period = 0
    end_period = 1
    num_iter = 0

    while x[-1] == -1:

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

    return x


if __name__ == '__main__':
	# Demo run displaying profiles.

	init_age = 16
	age_max = 96

	_, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
	for axis in axes.ravel():
		
		# Make synth screening history.
		x = simulate_profile(init_age, age_max)
		
		# Add profile to figure.
		plot_profile(x, axis, show=False)

	plt.show()
	