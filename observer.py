import generator as gen
import numpy as np

def get_gen_params():
    params = {}

    params['init_bound'] = (0, 10)  # initial slope (degrees) and intercept of bound
    params['max_step'] = 50 # maximum time step (time is discrete)
    params['rr_window'] = 'fixed' # 'fixed' / 'dynamic' length of window
    params['win_len'] = 100 # number of trials over which to estimate reward-rate
    params['rwd'] = 100 # reward if correct
    params['penalty'] = 0 # reward if incorrect

    params['update_method'] = 'grad' # 'grad': gradient descent; 'greedy': epsilon-greedy

    ### Parameters related to Gradient descent
    params['lr'] = 2 # learning rate for gradient descent
    params['del_slope'] = 0.5 # degrees (for numerical derivative)
    params['del_inter'] = 1 # for numerical derivative

    ### Parameters for Greedy search
    params['greedy_eps'] = 0.3 # Jump probability (if RR at new location is better)
    params['slope_bounds'] = [-60,20] # [min, max] of slopes (degrees)
    params['inter_bounds'] = [0,20] # [min, max] of intercept (threshold)


    return params


def compute_thresh_time(bound, max_step):
    ''' Compute a discrete value of threshold at every point of time based on
        slope and intercept of bound
        Note: only returns the top bound. Bottom bound will be symmetrical
    '''
    slope = bound[0]
    intercept = bound[1]
    time_vec = np.arange(0, max_step, step=1)
    thresh = np.floor(np.radians(slope) * time_vec + intercept) # y=mx+c
    return thresh


def get_window_len(bound=(0,4)):
    ''' If RR is estimated over a dynamic window, establish the length of window
    '''
    params = get_gen_params()
    return params['win_len'] # placeholder - just return the fixed length


def simulate_trial(bound):
    ''' Simulate a trial, generating a random walk, comparing accumulated
        evidence with bound and returning Reward & RT.
    '''
    params = get_gen_params()

    ### Generate a sample random walk for a state of the world
    (true_state, walk) = gen.sample_trial()

    ### Get threshold as a function fo time for given bound
    positive_thresh = compute_thresh_time(bound, params['max_step'])
    negative_thresh = -1 * positive_thresh # symmetrical along x-axis

    ### Compare walk with positive and negative thresholds
    ### Assumes walk & threshold vectors are same length - catch exception (TBD)
    above_top = np.where(positive_thresh <= walk)[0] # index [0] because first (only) dimension of nparray
    below_bottom = np.where(negative_thresh >= walk)[0]
    if above_top.size == 0:
        rt_top = params['max_step']
    else:
        rt_top = above_top[0] # the first passage - top threshold
    if below_bottom.size == 0:
        rt_bottom = params['max_step']
    else:
        rt_bottom = below_bottom[0] # the first passage - bottom threshold

    ### Compute decision based on first passage
    if rt_top < rt_bottom:
        decision = 1 # accumulated evidence touched top bound first
        rt = rt_top
    elif rt_top > rt_bottom:
        decision = -1 # evidence touched bottom bound first
        rt = rt_bottom
    else: # both at same time
        decision = (2 * np.random.randint(0, 2)) - 1 # flip a coin
        rt = rt_top # doesn't matter which one

    walk = walk[0:rt+1] # clip the walk (doesn't matter what happens after threshold)

    ### Determine reward based on whether decision is correct or incorrect
    if decision == true_state:
        rwd = params['rwd']
    else:
        rwd = params['penalty']

    return (walk, decision, true_state, rwd, rt)


def estimate_rr(bound):
    ''' Estimate a reward rate for a particular bound by observing some random walks,
        making decisions and observing outcomes.
    '''
    params = get_gen_params()
    if params['rr_window'] == 'dynamic':
        win_len = get_window_len(bound)
    else:
        win_len = params['win_len']
    
    cum_rwd = 0 # initialise cumulative reward
    cum_time = 0 # initialise cumulative time
    walks = []
    decisions = []
    true_states = []
    for ii in range(win_len):
        (walk, decision, true_state, rwd, rt) = simulate_trial(bound)
        if rwd == params['rwd']:
            iti = gen.get_iti(correct=True)
        else:
            iti = gen.get_iti(correct=False)
        cum_rwd += rwd
        cum_time += (rt + iti)
        walks.append(walk)
        decisions.append(decision)
        true_states.append(true_state)

    rr = cum_rwd / cum_time

    return (rr, walks, decisions, true_states)


def gradient_descent(current_bound):
    ''' A local update method. Determines the gradient at any given location (averaged over some trials).
        Then jumps to a new bound based on gradient and learning rate
    '''
    params = get_gen_params()
    lr = params['lr']

    ### Numerically estimate gradient wrt slope and intercept
    ### df/dx = lim h-> 0 [f(x+h) - f(x) / h]
    ### Here we don't take limit for rough estimate -- otherwise will take very large number of trials
    (rr_curr, walks_curr, decisions_curr, true_states_curr) = estimate_rr(current_bound)
    del_slope = params['del_slope']
    del_inter = params['del_inter']
    new_slope_bound = (current_bound[0] + del_slope, current_bound[1])
    new_inter_bound = (current_bound[0], current_bound[1] + del_inter)
    rr_new_slope = estimate_rr(new_slope_bound)[0]
    rr_new_inter = estimate_rr(new_inter_bound)[0]
    dslope = (rr_new_slope - rr_curr) / del_slope # partial derivative for slope
    dinter = (rr_new_inter - rr_curr) / del_inter

    ### Compute new bound: lr * df/dx
    new_slope = current_bound[0] + (lr * dslope)
    new_inter = current_bound[1] + (lr * dinter)
    if new_inter <= 0:
        new_inter = 0 # lower bound on intercept is zero
    new_bound = (new_slope, new_inter)

    # grad = (dslope, dinter)
    return [new_bound, walks_curr, decisions_curr, true_states_curr, rr_curr]


def greedy_search(current_bound, rr_seq):
    ''' Sample new location with probability greedy_eps. If RR is better at new location, then jump.
        rr_seq = a list of RRs experienced at current location
    '''
    params = get_gen_params()

    ### Estimate RR at current location
    (rr_curr, walks_curr, decisions_curr, true_states_curr) = estimate_rr(current_bound)
    rr_seq.append(rr_curr) # Note: rr_seq is a global parameter so will be updated globally
    rr_curr = sum(rr_seq) / len(rr_seq) # keep averaging over all RRs experienced at current location

    ### with probability greedy_eps explore a new location
    if np.random.random_sample() <= params['greedy_eps']:
        ### Determine new location, by sampling from a uniform distribution over bound interval
        new_slope = np.random.randint(params['slope_bounds'][0], params['slope_bounds'][1]) 
        new_inter = np.random.randint(params['inter_bounds'][0], params['inter_bounds'][1]) 
        jump_bound = (new_slope, new_inter)

        ### If new RR is more than current RR, jump wp greedy_eps
        (rr_new_bound, walks_new_bound, decisions_new_bound, true_states_new_bound) = estimate_rr(jump_bound)
        
        if rr_new_bound > rr_curr:
            new_bound = jump_bound # update the location
            rr_seq = [rr_new_bound] # re-initialise the list
            # update the variables for the walks & RR so they are reflected in the figure
            rr_curr = rr_new_bound 
            walks_curr = walks_new_bound
            decisions_curr = decisions_new_bound
            true_states_curr = true_states_new_bound
        else:
            new_bound = current_bound
    else:
        new_bound = current_bound

    return [new_bound, walks_curr, decisions_curr, true_states_curr, rr_curr, rr_seq]


def update_bound(current_bound, rr_seq = []):
    ''' Given a bound, compute a new bound, with the goal of (eventually) getting
        to the optimal bound
        current_bound = (slope, intercept) of current bound
        rr_seq = a list of RR experienced at current location (used in eps-Greedy)
    '''

    params = get_gen_params()

    if params['update_method'] == 'grad':
        update = gradient_descent(current_bound)
    elif params['update_method'] == 'greedy':
        update = greedy_search(current_bound, rr_seq)

    return update
