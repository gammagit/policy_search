import numpy as np

def get_params():
    params = {}

    params['eps'] = 0.2 # range [0,0.5]; 0 = random, 0.5 = deterministic
    params['prwd'] = 1 # The probability that the correct state leads to reward
    params['length'] = 50 # Maximum length of random walk
    params['iti_c'] = 15 # Inter-trial interval for correct decision
    params['iti_in'] = 50 # Inter-trial interval for incorrect decision

    return params

def sample_trial():
    ''' On each trial, there is a ground-truth state (e.g. Left / Right) and
        a set of samples based on drift. This function randomly samples a true
        state and then generates a random walk based on params['eps']
    '''
    params = get_params()

    ### Get drift based on whether true state is +1 or -1
    true_state = (2 * np.random.randint(0,2)) - 1 # Sample true state - assumes both states equally likely
    if true_state == 1:
        drift = 0.5 + params['eps']
    elif true_state == -1:
        drift = 0.5 - params['eps']

    events = np.random.binomial(n=1, p=drift, size=params['length']) # Generates array [0, 1, 1,....  # Generates array [0, 1, 1,.... ] etc
    obs = 2 * events - 1 # Observations: convert events into +1, -1
    walk = np.cumsum(obs) # cumulative sum of observations

    return (true_state, walk)


def get_iti(correct = True):
    params = get_params()
    if correct == True:
        return params['iti_c']
    else:
        return params['iti_in']