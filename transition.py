import numpy as np

from probas import p_init_state


SEED = 0



def shift_from_s1(time_index, df_Pt, seed=None, states=[1, 2]):
    """Two events: P not state shift = 1 - P state shift."""

    if seed is None:
        np.random.seed(SEED)
    else:
        np.random.seed(seed)
    
    p2 = float(df_Pt.iloc[0, time_index])
    p1 = 1.0 - p2
    
    return int(np.random.choice(states, p=(p1, p2)))


def shift_from_s2(time_index, df_Pt, seed=None, states=[1, 2, 3]):
    """Three events where P sums to one."""

    if seed is None:
        np.random.seed(SEED)
    else:
        np.random.seed(seed)
    
    p3 = float(df_Pt.iloc[1, time_index])
    p1 = float(df_Pt.iloc[3, time_index])
    p2 = 1 - p1 - p3
    
    return int(np.random.choice(states, p=(p1, p2, p3)))


def shift_from_s3(time_index, df_Pt, seed=None, states=[2, 3, 4]):
    """Three events where P sums to one."""

    if seed is None:
        np.random.seed(SEED)
    else:
        np.random.seed(seed)
    
    p4 = float(df_Pt.iloc[2, time_index])
    p2 = float(df_Pt.iloc[4, time_index])
    p3 = 1 - p4 - p2
    
    return int(np.random.choice(states, p=(p2, p3, p4)))
