import numpy as np 

from utils import lambda_sr, age_partitions, age_group_idx
from transition import legal_transitions


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

    # NB: Adjust to Python counting logic.
    kappas = [kappa(current_age, current_state, t, i) for i in range(n + 1)]
    
    return 1.0 - np.exp(sum(kappas))


def sojourn_time_cdf(start_age, stop_age, current_state):

	# NOTE: Adjust to Python counting logic.
    time_lapse = int(stop_age - start_age + 1)

    cdf = np.zeros(time_lapse, dtype=np.float32)
    
    k = age_group_idx(start_age)
    for t in range(1, time_lapse):
        
        l = age_group_idx(start_age + t)
        
        cdf[t] = cumul_sojourn_time(l - k, start_age, current_state, t)

    return np.array(cdf)


def sojourn_time(start_age: int, age_max: int, current_state: int, seed: int = 0) -> float:
    """Estimate the time spent in a given state.

    Args:
        sojourn_cdf: Cumulative 
        
    Returns:
        The amount of time a female spends in the current state.
    """
    
    if current_state == 0:
        return age_max

    sojourn_cdf = sojourn_time_cdf(start_age, age_max, current_state)

    #np.random.seed(seed)
    u = np.random.uniform(low=0.0, high=1.0)
    
    t_lower = np.squeeze(np.where(u > sojourn_cdf))
    if np.ndim(t_lower) > 0:
        t_lower = t_lower[-1]

    l = age_group_idx(start_age + t_lower)
    k = age_group_idx(start_age)

    # NB: Adjust to Python counting logic.
    s = sum([kappa(start_age, current_state, start_age + t_lower, i) for i in range(1, l - k + 1)])
    scale = sum(legal_transitions(current_state, lambda_sr[l, :]))

    return (s - np.log(1 - u)) / scale
