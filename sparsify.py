import numpy as np

from .utils import age_group_idx


def sparse_profle(x, start_age, max_age, stepsize=3, missing=0):

	# Keep obs every stepsize years.
	if isinstance(stepsize, int):

		to_keep = np.arange(start_age, max_age, 1)[::stepsize]
		
		x_sparse = np.ones_like(x) * missing
		x_sparse[to_keep] = x[to_keep]

	return x_sparse


if __name__ == '__main__':
	import numpy as np
	x = np.arange(81)

	print(sparsen(x, 0, 80, 3))
