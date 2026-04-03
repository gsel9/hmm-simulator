"""Utilities for subsampling dense simulated screening histories.

Simulates real-world irregular screening by keeping only a subset of
timepoints from a fully-observed profile matrix.
"""

import numpy as np

from .utils import sample_start_age, sample_end_age


def sample_screenings(
    X, stepsize, proba_init_age=None, proba_dropout=None, missing=0
):
    n_timepoints = X.shape[1]

    X_sparse = []
    for x in X:
        if proba_init_age is None:
            min_age = np.argmax(x)
        else:
            min_age, min_age_idx = sample_start_age(
                n_timepoints, proba_init_age
            )

        if proba_dropout is None:
            max_age = np.argmax(np.cumsum(x))
        else:
            max_age = sample_end_age(
                n_timepoints, proba_dropout, min_age_idx
            )

        assert min_age < max_age + 1

        to_keep = np.arange(min_age, max_age, 1)[::stepsize]

        x_sparse = np.zeros_like(x)
        x_sparse[to_keep] = x[to_keep]

        if sum(x[to_keep]) == 0:
            continue

        X_sparse.append(x_sparse)

    return np.array(X_sparse)
