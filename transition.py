"""
"""

from typing import List, Union

import numpy as np

from .utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int) -> int:
    """Sample the state at first screening."""
    
    age_grp = age_group_idx(init_age)

    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_grp])       


def legal_transitions(current_state: int, lambdas: List, norm: bool = False) -> np.ndarray:
    """Extract transition intensities for the enabled state shifts given the 
    current state.  

    Args:
        current_state:
        lambdas: Transition intensities for a given age group.
        norm: Scale transition intensities to sum to one.

    Returns:
        Transition intensities relevant for the current state.
    """
    
    # Censoring.
    if current_state == 0:
        return
    
    # s1 -> s2 or s1 -> censoring.
    if current_state == 1:
        l_sr = [lambdas[0], lambdas[5]]
    
    # s2 -> s3 or s2 -> s1 or -> censoring.
    if current_state == 2:
        l_sr = [lambdas[1], lambdas[3], lambdas[6]]
    
    # s3 -> s4 or s3 -> s2 or -> censoring.
    if current_state == 3:
        l_sr = [lambdas[2], lambdas[4], lambdas[7]]

    # s4 -> s1 or s4 -> censoring.
    if current_state == 4:
        l_sr = [1 - lambdas[8], lambdas[8]]

    if not norm:
        return np.array(l_sr)

    return np.array(l_sr) / sum(l_sr)


def next_state(age: int, current_state: int, censoring: int = 0) -> int:
    """Simulate the next state from sojourn time conditions.

    Args:
        age:
        current_state: 
        censoring: Representation of censoring.

    Returns:
        The next state.
    """

    p = legal_transitions(current_state, lambda_sr[age_group_idx(age)], norm=True)

    # s1 -> s2 or s1 -> censoring.
    if current_state == 1:
        return np.random.choice((2, censoring), p=p)

    # s2 -> s3 or s2 -> s1 or -> censoring.
    if current_state == 2:
        return np.random.choice((3, 1, censoring), p=p)
    
    # s3 -> s4 or s3 -> s2 or -> censoring.
    if current_state == 3:
        return np.random.choice((4, 2, censoring), p=p)
    
    # s4 -> s1 or s4 -> censoring.
    if current_state == 4:
        return np.random.choice((1, censoring), p=p)

    return censoring


if __name__ == '__main__':
    next_state(16, 2)
