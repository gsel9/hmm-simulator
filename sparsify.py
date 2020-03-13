import numpy as np

from utils import age_group_idx


def sparsen_profle(x, start_age, max_age, stepsize=3, missing=0):

	# Keep obs every stepsize years.
	if isinstance(stepsize, int):

		to_keep = np.arange(start_age, max_age, 1)[::stepsize]
		
		x_sparse = np.ones_like(x) * missing
		x_sparse[to_keep] = x[to_keep]

	return x_sparse


# DEPRECATED
def clip_init_observation(x_sparse, init_age, age_max, proba_init_age, missing, return_idx=False):
	"""Determine time point of first observation."""

	# Define time grid with same resolution as PDF.
	ages = np.linspace(init_age, age_max, len(proba_init_age))

	init_age_idx = int(np.random.choice(range(len(ages)), p=proba_init_age))

	start_age = int(ages[init_age_idx])
	x_sparse[:start_age] = missing

	if return_idx:
		return init_age_idx


# DEPRECATED
def clip_final_observation(x_sparse, init_age, age_max, proba_dropout, missing, init_age_idx):
	"""Determine time point of last observation."""
	
	# Define time grid with same resolution as PDF.
	ages = np.linspace(init_age, age_max, len(proba_dropout))

	# Truncate PDF to consider only time points > inital observation.
	p = proba_dropout[init_age_idx:] / sum(proba_dropout[init_age_idx:])
    
	end_age = int(np.random.choice(ages[init_age_idx:], p=p))
	x_sparse[end_age:] = missing


if __name__ == '__main__':
	import numpy as np
	x = np.arange(81)

	print(sparsen(x, 0, 80, 3))
