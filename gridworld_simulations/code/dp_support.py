import numpy as np
import itertools
from scipy.optimize import minimize

def interpolate_V(V,state_next,alpha_next,alpha_set,t,debug=False):
    '''Interpolates CVaR value function between alpha levels.
    '''

    assert alpha_set[0]==0.0 # make sure 0 is part of interpolation

    alpha_next_i_nearest = np.argmin(np.abs(alpha_set-alpha_next))
    alpha_next_nearest = alpha_set[alpha_next_i_nearest]

    if alpha_next>1:
        # return highest alpha
        return(V[state_next,len(alpha_set)-1,t])
    elif alpha_next<0:
        # shouldn't be able to happen, so break if it does
        import pdb; pdb.set_trace()
    elif alpha_next==alpha_next_nearest:
        # no need for interpolation, so skip calculations
        return(V[state_next,alpha_next_i_nearest,t])
    else:
        # find lower and upper y_nearest.
        if alpha_next_nearest<alpha_next:
            alpha_next_i_upper = alpha_next_i_nearest+1
            alpha_next_upper = alpha_set[alpha_next_i_upper]
            alpha_next_i_lower = alpha_next_i_nearest
            alpha_next_lower = alpha_next_nearest
        elif alpha_next_nearest>alpha_next:
            alpha_next_i_upper = alpha_next_i_nearest
            alpha_next_upper = alpha_next_nearest
            alpha_next_i_lower = alpha_next_i_nearest-1
            alpha_next_lower = alpha_set[alpha_next_i_lower]

        # calculate slope
        slope = (V[state_next,alpha_next_i_upper,t] - V[state_next,alpha_next_i_lower,t]) / (alpha_next_upper - alpha_next_lower)

        # interpolate: start at lower, add difference times the slope
        V_interp = V[state_next,alpha_next_i_lower,t] + slope*(alpha_next-alpha_next_lower)

        return(V_interp)

def distorted_value_objective_fun(dist_weights,
                                  next_state_reward_pairs,
                                  prob_next_state_reward_pairs,
                                  V,
                                  alpha,
                                  alpha_i,
                                  alpha_set,
                                  t,
                                  gamma,
                                  cvar_type='pCVaR',
                                  debug=False):
    '''Function that is minimized when calculating the distorted expectation for each (state, action) pair.
    '''

    if np.any(np.isnan(dist_weights)):
        return(np.inf)

    distorted_exp=0.0

    # loop over next states and rewards, associated probabilities, and distortion weights
    for (next_state,reward),prob,dweight in zip(next_state_reward_pairs,
                                                prob_next_state_reward_pairs,
                                                dist_weights):

        if cvar_type=='pCVaR' or cvar_type=='fCVaR':
            Vp=interpolate_V(V,next_state,alpha*dweight,alpha_set,t+1) # calculate interpolate value function
            distorted_exp += prob*dweight*(reward + gamma*Vp) # no multiplication by distortion weight (because of interpolation)
        elif cvar_type=='nCVaR':
            Vp=V[next_state,alpha_i,t+1] # nCVaR doesn't require intepolation, so access value table directly
            distorted_exp += prob*dweight*(reward + gamma*Vp)

    return(distorted_exp)

def Q_backup(next_states,
            p_next_states,
            rewards,
            p_rewards,
            alpha,
            alpha_i,
            V,
            t, # time-step
            cvar_type,
            gamma,
            alpha_set,
            max_inner_iters=50,
            multi_starts_N=10,
            same_answer_ns=3,
            same_answer_tol=1e-4):

        '''Calculate CVaR Q-values for a single state and alpha level.

        Requires a list of next possible states, rewards, and associated probabilities.
        '''

        # set up pairs of possible next states and rewards (and their probabilities)
        next_state_reward_pairs = [pair for pair in itertools.product(next_states,rewards)]
        prob_next_state_reward_pairs = [probs[0]*probs[1] for probs in itertools.product(p_next_states,p_rewards)] # indedendent so multiply

        # do worst-case outcome for alpha=0
        if alpha == 0:

            # find minimum next state value
            minV = np.min(V[next_states,0,t+1])

            # find the minimum reward
            minr = np.min(rewards)

            # calculate current value
            Q = minr + gamma*minV

            dist_weights = np.zeros((len(rewards),len(next_states)))*np.nan
            success = True
            return(Q,dist_weights,success)

        # for all other values of alpha
        else:

            # bounds for distortion weights, which depends on current alpha
            bnds = tuple(((0.0,1.0/alpha) for _ in range(len(prob_next_state_reward_pairs))))

            # sum to 1 constraint for distortion weights
            def sum_to_1_constraint(dist_weights):
                zero = np.dot(dist_weights,np.array(prob_next_state_reward_pairs))-1 # p_trans is found in one python env up
                return(zero)

            cons = ({'type': 'eq', 'fun': sum_to_1_constraint})
            succeeded_at_least_once = False
            results_list = []
            fun_mins = []

            # repeat the distorted expectation calculation multiple times
            for _ in range(max_inner_iters):

                # distortion weight initial values
                n_probs = len(prob_next_state_reward_pairs)

                # initial values: uniform in probability simplex
                dist_probs_init = np.random.dirichlet(np.ones(n_probs),size=1)[0]
                dist_weights_init = dist_probs_init/np.array(prob_next_state_reward_pairs)
                assert np.abs(sum_to_1_constraint(dist_weights_init))<1e-4

                results = minimize(distorted_value_objective_fun,
                                   dist_weights_init,
                                   args=(next_state_reward_pairs,prob_next_state_reward_pairs,
                                   V,alpha,alpha_i,alpha_set,t,gamma,cvar_type,False),
                                   method='SLSQP',
                                   bounds=bnds,
                                   constraints=cons)

                if results.success:
                    succeeded_at_least_once=True
                    results_list.append(results)
                    fun_mins.append(results.fun)

                # exit early if all N minimums are the same
                if len(fun_mins)>same_answer_ns:
                    minn = np.min(np.array(fun_mins))
                    num_within_tol = np.sum((np.array(fun_mins)-minn)<same_answer_tol)
                    if num_within_tol>=same_answer_ns:
                        break

                # or exit after max number of multi-starts have been exceeded
                if len(fun_mins)>multi_starts_N:
                    break

            # find minimum over multi-starts
            argmin_funs = np.argmin(np.array(fun_mins))
            results = results_list[argmin_funs]

            # unpack results
            dist_weights = results.x
            Q = results.fun
            success = results.success
            if success==False:
                print('Failed')

            return(Q,dist_weights,success)

def Q_backup_horizon(rewards,
                     p_rewards,
                     alpha,
                     alpha_set):
    '''Calculate CVaR Q-values for a single state and alpha level at the planning horizon (i.e. t=T).

    Only requires a list of next rewards and associated probabilities.
    '''

    assert type(rewards)==np.ndarray
    assert len(rewards)==len(p_rewards)

    if alpha==0.0:

        minR = np.min(rewards)
        minR_idcs = np.where(rewards==minR)[0]

        Q = minR
        dist_weights = np.zeros(len(p_rewards))
        dist_weights[minR_idcs]=1/(p_rewards[minR_idcs]*len(minR_idcs)) # take the weight as 1/prob, but if there are more than 1 additionally divide by total)

        assert np.abs((dist_weights*p_rewards).sum()-1)<0.01

        success=True
        return(Q,dist_weights,success)

    else:

        def obj_fun(dist_weights,p_rewards,rewards):
            return(np.sum((dist_weights*p_rewards*rewards)))

        def sum_to_1_constraint(dist_weights):
            return(np.dot(dist_weights,p_rewards)-1)

        bnds = tuple(((0.0,1.0/alpha) for _ in range(len(p_rewards))))
        dist_weights_init = np.random.uniform(alpha_set[1],1.0/alpha,len(rewards))
        cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

        results = minimize(obj_fun,
                           dist_weights_init,
                           args=(p_rewards,rewards),
                           method='SLSQP',
                           bounds=bnds,
                           constraints=cons)

        dist_weights = results.x
        Q = results.fun
        success = results.success

        assert np.abs((1-np.dot(dist_weights,p_rewards)))<0.01
        assert np.all(dist_weights<=(1.0/alpha+0.01))
        assert np.all(0<=dist_weights)

        return(Q, dist_weights, success)

def Update_Q_Values(s,
                    t,
                    V_CVaR,
                    Nalpha,
                    Na,
                    Nr,
                    Ns,
                    T,
                    alpha_set,
                    task,
                    cvar_type,
                    gamma,
                    same_answer_ns):
    '''Update CVaR Q-values for all actions and alphas at a single state.

    This wraps the `Q_backup` function, looping over actions and alphas.
    '''

    Q_CVaR_tmp = np.zeros((Na,Nalpha))
    Xis_tmp = np.zeros((Na,Nalpha,Nr,Ns))

    # loop over possible alphas
    alphas_to_iterate = range(Nalpha)
    for alpha_i in alphas_to_iterate:

        alpha = alpha_set[alpha_i]

        # get actions to loop over
        try:
            actions_to_iterate = task.actions_allowed_in_state(s)
        except:
            actions_to_iterate = range(Na)

        for a in actions_to_iterate:

            # update CVaR Q-value at horizon
            if t==(T-1):

                # get possible rewards current state
                non_zero_reward_idcs = np.where(task.p_r[s,:]!=0.0)[0] # where probability is not zero
                rewards = task.r_support[non_zero_reward_idcs]
                p_rewards = task.p_r[s,non_zero_reward_idcs]

                Q_CVaR_tmp[a,alpha_i],xis,success = Q_backup_horizon(np.array(rewards),
                                                         np.array(p_rewards),
                                                         alpha,
                                                         alpha_set)

                Xis_tmp[a,alpha_i,non_zero_reward_idcs,:]=np.tile(xis[:,np.newaxis],Ns)

            else:

                # get possible rewards current state
                non_zero_reward_idcs = np.where(task.p_r[s,:]!=0.0)[0] # where probability is not zero
                rewards = task.r_support[non_zero_reward_idcs]
                p_rewards = task.p_r[s,non_zero_reward_idcs]

                # get next states with non-zero transition prob
                next_states = np.where(task.P[s,:,a]!=0.0)[0]
                p_next_states = task.P[s,next_states,a]

                if len(next_states)==0:
                    import pdb; pdb.set_trace()

                # do Q-value back-up
                Q_CVaR_tmp[a,alpha_i],xis,success = Q_backup(next_states,
                                                             p_next_states,
                                                             rewards,
                                                             p_rewards,
                                                             alpha,
                                                             alpha_i,
                                                             V_CVaR,
                                                             t,
                                                             cvar_type,
                                                             gamma,
                                                             alpha_set,
                                                             same_answer_ns=same_answer_ns)

                # store Xis over next states and possible rewards
                try:
                    Xis_tmp[a,alpha_i,non_zero_reward_idcs,next_states]=np.squeeze(xis)
                except:
                    Xis_tmp[a,alpha_i,non_zero_reward_idcs,next_states]=np.nan

    return Q_CVaR_tmp,Xis_tmp
