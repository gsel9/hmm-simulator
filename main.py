from typing import Union, List

import numpy as np

from transition import next_state, inital_state
from sojourn import sojourn_time 


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

	# Simulate five females over 80 years (16-96 yo).
	num_females = 5
	init_age = 16
	age_max = 94

	for _ in range(num_females):
		x = simulate_profile(init_age, age_max)
		print(x)

	# NOTE:
	# Modify `age_partitions` in `utils.py` to adjust the resolution 
	# of the time domain.
