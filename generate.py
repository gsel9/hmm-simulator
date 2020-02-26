import numpy as np 

from transition import set_init_state, sample_scores


def data_generator(num_samples, num_tpoints, missing=0):
	
    X = np.ones((num_samples, num_tpoints), dtype=np.int32) * missing

    # Initial female condition.
    X, idx = set_init_state(X, return_idx=True)

    for num in range(num_samples):

        # Next time points.
        next_screen = idx + num + 1

        # Valid time points < num_tpoints.
        idx = next_screen[next_screen < num_tpoints + 1]

        X = sample_scores(X, idx)

    return X


def set_init_state(X, seed=None, return_idx=False):

    # Time of first screening.
    init_idx = init_idx(num, seed=seed)
    
    # State at initial screening.
    X[init_idx] = init_state(num, seed=seed)

    if return_idx:
        return X, init_idx

    return X


def sample_scores(X, idx):

    prev_states = X[idx]

        




def init_state(x, time_index, df_init):
    """Assings intial state according to empirical probabilities of being in a
    particular state at the time of the first screening."""

    if seed is None:
        np.random.seed(SEED)
    else:
        np.random.seed(seed)
    
    new_state = np.random.choice([1, 2, 3, 4], p=df_init.iloc[:, time_index])
    
    x[time_index] = int(new_state)



def sample_scores(X, num_tpoints):

    pass
 

def _impute_trajectory(x, otrain, Ptransition, force_init=False, break_at_dropout=True):
    """
    Args:
        x: A row in X.
        otrain: A row in Otrain.
    """

    # Impute x[0] with a randomly selected value.
    if force_init:
        set_init_state(x, time_index=0)

    if break_at_dropout:
        nz = x.nonzero()[0]
    else:
        nz = np.append(x.nonzero()[0], len(x))

    for num, nz_idx in enumerate(nz[:-1]):
        
        next_nz_idx = int(nz[num + 1])
        
        # No need to impute.
        if next_nz_idx - nz_idx < 2:
            continue

        prev_state = int(x[nz_idx])
        for time_index in range(nz_idx + 1, next_nz_idx):
            
            if prev_state == 1:
                state = shift_from_s1(time_index, Ptransition)
            
            elif prev_state == 2:
                state = shift_from_s2(time_index, Ptransition)
            
            elif prev_state == 3:
                state = shift_from_s3(time_index, Ptransition)

            else:
                continue

            x[time_index] = state
            otrain[time_index] = 1

            prev_state = state


def impute(X, O_train, O_test):

    p_transit = get_transition_probas()

    X_imp = X.copy()
    O_imp = O_train.copy()

    start_idx = np.argmax(O_train, axis=1)
    end_idx = np.argmax(np.cumsum(O_train, axis=1), axis=1)

    for num, (x, o) in enumerate(zip(X_imp, O_imp)):
        impute_trajectory(x, o, start_idx[num], end_idx[num], p_transit)

    X_imp[O_train.nonzero()] = X[O_train.nonzero()]
    X_imp[O_test.nonzero()] = X[O_test.nonzero()]

    return X_imp, O_imp


def impute_trajectory(x, o, start_idx, end_idx, p_transit, missing=0):

    idx = start_idx
    while idx < end_idx:

        # Propagate zeros after cancer cases.
        current_score = x[idx]
        if current_score == 0:
            idx = idx + 1

            continue

        next_score = x[idx + 1]
        if next_score == missing:
            x[idx + 1] = sample_next_score(current_score, idx + 1, p_transit=p_transit)

        idx = idx + 1

    o[start_idx:end_idx] = 1


def sample_next_score(current_score, next_idx, p_transit, num_iter=10):

    np.random.seed(0)

    scores = []
    for _ in range(num_iter):

        if current_score == 1:
            next_state = shift_from_s1(next_idx, p_transit)

        elif current_score == 2:
            next_state = shift_from_s2(next_idx, p_transit)

        elif current_score == 3:
            next_state = shift_from_s3(next_idx, p_transit)

        else:
            # Propagate zeros after cancer cases.
            next_state = 0

        scores.append(next_state)

    return np.random.choice(scores)