from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from transition import next_state, inital_state
from sojourn import sojourn_time 


def plot_profile(x, title=None, show=True, path_to_fig=None):

	plt.figure(figsize=(12, 8))

	if title is not None:
		plt.title(title, fontsize=22)

	plt.plot(x, '-o')
	plt.yticks(range(1, 5), range(1, 5), fontsize=18)
	plt.xticks(np.linspace(0, 80, 6, dtype=int), np.linspace(16, 96, 6, dtype=int), fontsize=18)
	plt.ylabel('State', fontsize=20)
	plt.xlabel('Years', fontsize=20)

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

    x = np.zeros(int(age_max - init_age), dtype=np.int32)

    # Set state at first screening.
    current_state = inital_state(init_age=init_age)
    
    current_age = init_age
    start_period = 0
    end_period = 1
    
    while x[int(age_max - init_age - 1)] == 0:
    
        # Time spent in current state.        
        dt = sojourn_time(current_age, age_max, current_state)
        
        end_period = end_period + int(dt)
        
        # Update profile values with current state.
        x[start_period:end_period] = current_state

        current_age = current_age + int(dt)
        current_state = next_state(current_age, current_state=current_state)
        
        start_period = end_period

    return x


if __name__ == '__main__':

	# Simulate two females over 80 years (16-96 yo).
	num_females = 2
	init_age = 16
	age_max = 96

	for num in range(num_females):
		
		# Make synth screening history.
		x = simulate_profile(init_age, age_max)

		# Display each synth profile.
		plot_profile(x, show=True, title=f"Synthetic profile: {num + 1}")
	
	# NOTE:
	# Modify `age_partitions` in `utils.py` to adjust the resolution 
	# of the time domain.
