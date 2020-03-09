from typing import List, Union

import numpy as np

from utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int, seed: int = 0):
    """Sample state at first screening.
    
    Args:
        x:
        
    Returns:
        
    """
    
    #np.random.seed(seed)
    
    age_grp = age_group_idx(init_age)
                  
    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_grp])       


def legal_transitions(current_state: int, lambdas: List) -> List:
    
    # TEMP:
    #return lambdas
    
    # s1 -> s2 or s1 -> death/dropout.
    if current_state == 1:
        return [lambdas[0], lambdas[5]]
    
    # s2 -> s3 or s2 -> s1 or -> death/dropout.
    if current_state == 2:
        return [lambdas[1], lambdas[3], lambdas[6]]
    
    # s3 -> s4 or s3 -> s2 or -> death/dropout.
    if current_state == 3:
        return [lambdas[2], lambdas[4], lambdas[7]]

    # s4 -> s1 or s4 -> death/dropout.
    return [1 - lambdas[8], lambdas[8]]


def next_state(age: int, current_state: int, death: int = 0, 
               seed: int = 0) -> int:
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

    lambdas = legal_transitions(current_state, lambda_sr[age_group_idx(age)])

    # s1 -> s2 or s1 -> death/dropout.
    if current_state == 1:
        return np.random.choice((2, death), p=lambdas / sum(lambdas))

    # s2 -> s3 or s2 -> s1 or -> death/dropout.
    if current_state == 2:
        return np.random.choice((3, 1, death), p=lambdas / sum(lambdas))
    
    # s3 -> s4 or s3 -> s2 or -> death/dropout.
    if current_state == 3:
        return np.random.choice((4, 2, death), p=lambdas / sum(lambdas))
    
    # s4 -> s1 or s4 -> death/dropout.
    if current_state == 4:
        return np.random.choice((1, death), p=lambdas / sum(lambdas))

    return death
