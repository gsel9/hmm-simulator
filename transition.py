from typing import List

import numpy as np

from utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int, seed: int = 0):
    """Sample state at first screening.
    
    Args:
        x:
        
    Returns:
        
    """
    
    np.random.seed(seed)
    
    age_grp = age_group_idx(init_age)
                  
    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_grp])       


# ERROR: Potentially something wrong with def of transit probas.
def legal_transitions(current_state: int, lambdas: List) -> List:
    
    # TEMP:
    return lambdas
    
    # s1 => s2
    if current_state == 1:
        return [lambdas[0]]
    
    # s2 => s3 or s2 => s1
    if current_state == 2:
        return [lambdas[1], lambdas[3]]
    
    # s3 => s4 or s3 => s2
    if current_state == 3:
        return [lambdas[2], lambdas[4]]


def next_state(age: int, current_state: int, seed: int = 0) -> int:
    """Simulate the next state from sojourn time conditions.

    Args:
        age:
        current_state: 
        seed: Reproduce the pseudo-random number generator.

    Returns:
        The next state.

    Note:
        * Consider only valid transitions.
        * Assume successful treatment if cancer (s4) and transits to normal state.
    """

    # s1 -> s2
    if current_state == 1:
        return 2

    # NB: Assume successful treatment and transits to normal state.
    if current_state == 4:
        return 1

    np.random.seed(seed)

    age_group = age_group_idx(age)

    # s2 -> s3 or s2 -> s1
    if current_state == 2:
        lambdas = [lambda_sr[age_group, 1], lambda_sr[age_group, 3]]
    
    # s3 -> s4 or s3 -> s2
    if current_state == 3:
        lambdas = [lambda_sr[age_group, 2], lambda_sr[age_group, 4]]
    
    # s2 -> s3 or s2 -> s1
    if current_state == 2:
        return np.random.choice((3, 1), p=lambdas / sum(lambdas))
    
    # s3 -> s4 or s3 -> s2
    if current_state == 3:
        return np.random.choice((4, 2), p=lambdas / sum(lambdas))
