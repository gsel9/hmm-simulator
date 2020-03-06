from typing import Union, List

import numpy as np

from generate import (
	sojourn_time_cdf, sojourn_time, 
	next_state, inital_state
)


# TEMP: Current start_period and end_period assumes x has 80 entries but can be scaled
# to increase resolution.
def simulate_profile(x: Union[List, np.ndarray], current_age=16, age_max=96):
    """Update the profile vector of a single female. 

    Args:
    	x: Female profile vector.
    	current_age:
		age_max:
    """

    # Set state at first screening.
    current_state = inital_state(init_age=current_age)
    
    start_period = 0
    end_period = 1
    
    while x[-1] == 0:
        
        # Time spent in current state.
        cdf = sojourn_time_cdf(current_age, age_max, current_state)
        
        dt = sojourn_time(cdf, current_age, current_state)
        
        end_period = end_period + int(dt)
        
        # Update profile values with current state.
        x[start_period:end_period] = current_state

        current_age = current_age + int(dt)
        current_state = next_state(current_age, current_state=current_state)
        
        start_period = end_period


if __name__ == '__main__':

	# Simulate five females over 80 years (16-96 yo)
	X = np.zeros((5, 80), dtype=np.int32)

	for x in X:
	    simulate_profile(x, current_age=16, age_max=96)

	print(X)
