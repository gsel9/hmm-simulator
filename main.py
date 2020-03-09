from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from transition import next_state, inital_state
from sojourn import sojourn_time 


def plot_profile(x, axis, title=None, show=True, path_to_fig=None):

	#plt.figure(figsize=(12, 8))

	if title is not None:
		plt.title(title, fontsize=22)
		
	axis.plot(x, '-o')

	axis.set_yticks(range(0, 5))
	axis.set_yticklabels(['D5', 'N0', 'L1', 'L2', 'C4'], fontsize=18)

	axis.set_xticks(np.linspace(0, 80, 6, dtype=int))
	axis.set_xticklabels(np.linspace(16, 96, 6, dtype=int), fontsize=18)
	
	#plt.yticks(range(1, 5), range(1, 5), fontsize=18)
	#plt.xticks(np.linspace(0, 80, 6, dtype=int), np.linspace(16, 96, 6, dtype=int), fontsize=18)
	axis.set_ylabel('State', fontsize=20)
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
    """

    x = np.ones(int(age_max - init_age)) * -1

    # Set state at first screening.
    current_state = inital_state(init_age=init_age)
    prev_state = current_state
    
    current_age = init_age
    start_period = 0
    end_period = 1
    
    safety = 0
    while x[-1] == -1:

    	# Time spent in current state.
    	dt = sojourn_time(current_age, age_max, current_state)

    	end_period = end_period + int(dt)
    	current_age = current_age + int(dt)

    	x[start_period:end_period] = current_state

    	prev_state = current_state

    	# Update profile values with current state.
    	current_state = next_state(age=current_age, current_state=current_state, death=0)

    	# Sanity check.
    	if current_state != 0:
	    	msg = 'current_state: {} prev_state: {}'
	    	assert abs(current_state - prev_state) <= 1, msg.format(current_state, prev_state)

    	start_period = end_period

    	# To avoid endless loop.
    	safety += 1
    	if safety > len(x):
    		raise RuntimeError('Endless loop')

    return x


if __name__ == '__main__':
	# TODO:
	# * CHECK indices, summing and iterations in all Eqs.
	# * CHECK transit intensities.
	# * Use simulate_profile(num_timepoints) and scale the ranges specified in age_partitions. 

	init_age = 16
	age_max = 96

	_, axes = plt.subplots(nrows=6, ncols=4, figsize=(15, 15))
	for num, axis in enumerate(axes.ravel()):
	#for _ in range(2):
		
		# Make synth screening history.
		x = simulate_profile(init_age, age_max)
		
		# Display each synth profile.
		plot_profile(x, axis, show=False)

	#plt.show()
	plt.savefig('/Users/sela/Desktop/hmm_synth_profiles.pdf')
	# NOTE:
	# Modify `age_partitions` in `utils.py` to adjust the resolution 
	# of the time domain.
