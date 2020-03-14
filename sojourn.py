"""
"""

import numpy as np 

from .utils import lambda_sr, age_partitions, age_group_idx
from .transition import legal_transitions


def kappa_0(init_age, current_state, t) -> float:

    l = age_group_idx(init_age + t)
    s = sum(legal_transitions(current_state, lambda_sr[l], norm=False))

    return -1.0 * t * s


def kappa_1(age, current_state, t) -> float:
    
    k = age_group_idx(age)
    tau_kp = age_partitions[k + 1][0]

    l = age_group_idx(age + t)
    tau_l = age_partitions[l][0]
    
    s_k = sum(legal_transitions(current_state, lambda_sr[k], norm=False))
    s_l = sum(legal_transitions(current_state, lambda_sr[l], norm=False))
    
    return -1.0 * (tau_kp - age) * s_k - (age - tau_l) * s_l
    

def kappa_m(age, current_state, m) -> float:
    
    k = age_group_idx(age)
    
    km = k + m
    tau_km = age_partitions[km][1]

    kmm = km - 1
    tau_kmm = age_partitions[kmm][0]

    s_kmm = sum(legal_transitions(current_state, lambda_sr[kmm, :], norm=False))
    
    return -1.0 * (tau_km - tau_kmm) * s_kmm


def kappa(init_age, current_state, t, i) -> float:
    
    if i == 0:
        return kappa_0(init_age, current_state, t)
    
    if i == 1:
        return kappa_1(init_age, current_state, t)
    
    return kappa_m(init_age, current_state, i)


def eval_sojourn_time(n: int, age: int, current_state: int, t: int) -> float:
    """Ealuate the CDF for the sojourn time at time t.
    
    Args:
        n: Number of additional age partitions covered by the time lapse `t`
            when starting at `age`.
        age:
        current_state:
        t: Time passed since `age`.
    
    Returns:
        The CDF at time t.
    """

    # NB: Adjust to Python count logic.
    kappas = [kappa(age, current_state, t, i) for i in range(n)]
    
    return 1.0 - np.exp(sum(kappas))


# NB: CDF does not accumulate to 1.
def sojourn_time_cdf(start_age, stop_age, current_state) -> np.ndarray:
    """Compute the sojourn time CDF for a given female.
    
    Returns:
        The CDF evaluated at times t from start_age to stop_age.
    """

    # NOTE: Adjust to Python counting logic.
    time_lapse = int(stop_age - start_age + 1)

    cdf = np.zeros(time_lapse, dtype=np.float32)

    k = age_group_idx(start_age)
    for t in range(1, time_lapse):
        
        l = age_group_idx(start_age + t)
        
        cdf[t] = eval_sojourn_time(l - k, start_age, current_state, t)

    return np.array(cdf)


def sojourn_time(start_age: int, age_max: int, current_state: int) -> float:
    """Estimate the time that will spent in a given state.
    Args:
        start_age: 
        age_max:  
        current_state: 
        
    Returns:
        The amount of time a female spends in the current state.
    """
    
    # Censor the rest of the profile.
    if current_state == 0:
        return age_max

    sojourn_cdf = sojourn_time_cdf(start_age, age_max, current_state)

    # Corollary 1, step 1.
    u = np.random.uniform(low=0.0, high=1.0)

    # Step 2.
    k = age_group_idx(start_age)

    # Step 3.
    t_lower = np.squeeze(np.where(u > sojourn_cdf))
    if np.ndim(t_lower) > 0:
        t_lower = t_lower[-1]

    # Shift time point by age to satisfy l: P(T < tau_l - a) < u.
    l = age_group_idx(start_age + t_lower)

    # NB: Adjust to Python count logic and start sum from 1.
    sum_k = sum([kappa(start_age, current_state, start_age + t_lower, i) for i in range(1, l - k)])
    sum_p = sum(legal_transitions(current_state, lambda_sr[l, :]))

    # Step 4.
    return (sum_k - np.log(1 - u)) / sum_p


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    cdf = sojourn_time_cdf(16, 96, 1)
    print(cdf)

    plt.figure()
    plt.plot(cdf)
    plt.show()
    
    #print(sojourn_time(16, 36, 1))
    #for a in np.linspace(16, 96, 5, int):
    #    for b in np.linspace(a, 96, 5, int):
    #        print(sojourn_time(a, b, 1))
            #print(a, b, sojourn_time(a, b, 1))
    
