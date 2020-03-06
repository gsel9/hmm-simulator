from typing import List

import numpy as np

from probas import lambda_sr, p_init_state


age_partitions = np.array([
    (16, 19),  
    (20, 24),
    (25, 29),
    (30, 34), 
    (35, 39), 
    (40, 49), 
    (50, 59), 
    (60, 96)
])


def inital_state(init_age: int, seed: int = 0):
    """Sample state at first screening.
    
    Args:
        x:
        
    Returns:
        
    """
    
    np.random.seed(seed)
    
    age_grp = age_group_idx(init_age)
                  
    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_grp])       


# TODO: Handle out of bounds age grouping.
def age_group_idx(age: int):
    """Returns index for the age group."""
    
    for num, (tau_p, tau_pp) in enumerate(age_partitions):
        
        if age in range(tau_p, tau_pp + 1):
            return num
        
    #raise ValueError(f'Could not find any group for age {age}')        
    return num


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


def kappa_0(init_age, current_state, t):

    l = age_group_idx(init_age + t)
    
    s = sum(legal_transitions(current_state, lambda_sr[l]))

    return -1.0 * t * s


def kappa_1(age, current_state, t):
    
    k = age_group_idx(age)
    tau_kp = age_partitions[k + 1][0]

    l = age_group_idx(age + t)
    tau_l = age_partitions[l][0]
    
    s_k = (age - tau_kp) * sum(legal_transitions(current_state, lambda_sr[k]))
    s_l = (tau_l - age) * sum(legal_transitions(current_state, lambda_sr[l]))
    
    return s_k + s_l
    

def kappa_m(age, current_state, m):
    
    k = age_group_idx(age)
    km = k + m
    kmm = km - 1
    
    tau_km = age_partitions[km][1]
    tau_kmm = age_partitions[kmm][0]

    s_kmm = sum(legal_transitions(current_state, lambda_sr[kmm, :]))
    
    return (tau_kmm - tau_km) * s_kmm


def kappa(init_age, current_state, t, m):
    
    if m == 0:
        return kappa_0(init_age, current_state, t)
    
    if m == 1:
        return kappa_1(init_age, current_state, t)
    
    return kappa_m(init_age, current_state, m)


def cumul_sojourn_time(n: int, current_age: int, current_state: int, t: int) -> float:
    """Cumulative distribution for the sojourn time.
    
    Args:
        n: Number of additional age partitinos covered by the time lapse `t`
            when starting at age `current_age`.
        current_age:
        current_state:
        t: Time passed since `current_age`.
    
    Returns:
        The cumulative distribution for the sojourn time evaluated at time t.
    """

    # NB: Range over n + 1 for additional age paritions <= 7 (as expected).
    kappas = [kappa(current_age, current_state, t, i) for i in range(n + 1)]
    
    return 1.0 - np.exp(sum(kappas))
    

def sojourn_time_cdf(start_age, stop_age, current_state):
    
    time_lapse = int(stop_age - start_age)

    cdf = np.zeros(time_lapse, dtype=np.float32)
    
    k = age_group_idx(start_age)
    for t in range(1, time_lapse):
        
        l = age_group_idx(start_age + t)
        
        cdf[t] = cumul_sojourn_time(l - k, start_age, current_state, t)

    return np.array(cdf)


# QUESTION: n = l - k + 1 or n = l - k?
def sojourn_time(sojourn_cdf: np.ndarray, start_age: int,
                 current_state: int, seed: int = 0) -> float:
    """Estimate the time spent in a given state.

    Args:
        sojourn_cdf: Cumulative 
        
    Returns:
        The amount of time a female spends in the current state.
    """

    np.random.seed(seed)
    u = np.random.uniform(low=0.0, high=1.0)
    
    t_lower = np.squeeze(np.where(u > sojourn_cdf))[-1]

    l = age_group_idx(start_age + t_lower)
    k = age_group_idx(start_age)
    print(l, k)
    # QUESTION: n = l - k + 1 or n = l - k? Using n = l - k + 1 gives less variance.
    s = sum([kappa(start_age, current_state, start_age + t_lower, i) for i in range(1, l - k + 1)])
    scale = sum(legal_transitions(current_state, lambda_sr[l, :]))

    return (s - np.log(1 - u)) / scale


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
