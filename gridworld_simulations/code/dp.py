import numpy as np
import multiprocessing
from multiprocessing import Pool

from dp_support import Update_Q_Values

def CVaR_DP(task, # MDP
            T=3, # planning horizon
            alpha0=0.3, # risk preference for start state
            alpha_set = np.array([0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0]), # interpolation set for alpha
            cvar_type='pCVaR', # type of cvar to use; options: pCVaR, fCVaR, nCVaR
            gamma=1.0, # temporal discounting
            policy_to_evaluate=None, # given a policy, can be used to calculate the value function (i.e. policy evaluation)
            Q_roundoff=4, # rounding of Q-values for selecting best action
            same_answer_ns=3, # hyperparameter for inner optimization of CVaR (allows early stopping if same answer obtained multiple times)
            parallel=False # run computations using multiprocessing tools; much faster
            ):
    '''Main function for calculating the cvar-optimal policies.

    Requires a 'task' object, which specifies the Markov Decision Process (MDP).
    Can also be used to evaluate a fixed policy, returning just the value function.
    For policy evaluation, only deterministic policies are currently implemented.

        See "Figure_6: Gridworld_Simulation.ipynb" for typical use.

    '''

    Ns = task.Ns
    Na = task.Na
    Nalpha = len(alpha_set)
    Nr = task.Nr
    alpha0_i = np.where(alpha_set==alpha0)[0]

    Q_CVaR = np.zeros((Ns,Na,Nalpha,T))
    Xis = np.zeros((Ns,Na,Nalpha,Nr,Ns,T))
    V_CVaR = np.zeros((Ns,Nalpha,T))
    pi = np.zeros((Ns,Na,Nalpha,T))

    #------------------------------------#
    # work backwards from last time step #
    #------------------------------------#

    for t in reversed(range(0,T)):

        print('t='+str(t))

        #--------------------------------#
        # Update Q-Values for each state #
        #--------------------------------#

        states_to_iterate = task.states_allowed_at_time(t)

        if parallel:
            with Pool(multiprocessing.cpu_count()) as p:
                map_results = p.starmap(Update_Q_Values, [(s,t,V_CVaR,Nalpha,Na,Nr,
                  Ns,T,alpha_set,task,cvar_type,gamma,same_answer_ns) for s in states_to_iterate] )
        else:
            map_results = []
            for s in states_to_iterate:
                Q_CVaR_tmp,Xis_tmp =  Update_Q_Values(s,t,V_CVaR,Nalpha,Na,Nr,
                                        Ns,T,alpha_set,task,cvar_type,gamma,same_answer_ns)
                map_results.append((Q_CVaR_tmp,Xis_tmp))

        # unpack
        for s in states_to_iterate:
            Q_CVaR[s,:,:,t] = map_results[s][0]
            Xis[s,:,:,:,:,t] = map_results[s][1]

        #--------------------------------#
        # Update state values and policy #
        #--------------------------------#

        states_to_iterate = task.states_allowed_at_time(t)
        for s in states_to_iterate: # I could also maybe just do states to consider each round..

            # loop over possible alphas
            alphas_to_iterate = range(Nalpha)
            for alpha_i in alphas_to_iterate:

                alpha = alpha_set[alpha_i]

                # calculate optimal policy
                if policy_to_evaluate is None:
                    if cvar_type=='fCVaR':

                        # restrict actions to those allowed in each state (given by task)
                        # not 100% vetted for all types of MDPs
                        actions_allowed = np.array(task.actions_allowed_in_state(s))

                        # No matter what alpha level you are in the loop, use the alpha0_i to determine policy
                        Q_CVaR[s,:,alpha0_i,t] = np.round(Q_CVaR[s,:,alpha0_i,t],Q_roundoff)
                        Q_best_alpha0 = np.max(Q_CVaR[s,actions_allowed,alpha0_i,t])
                        best_actions = np.where(np.squeeze(Q_CVaR[s,:,alpha0_i,t])==Q_best_alpha0)[0]

                        filter=np.isin(best_actions,actions_allowed)
                        best_actions = best_actions[filter]

                        # (implicit tie-breaker to choose the first option)
                        # usually its a single action anyway
                        best_action = best_actions[0]

                        # set policy for current alpha level in loop
                        pi[s,:,alpha_i,t]=0.0
                        pi[s,best_action,alpha_i,t]=1.0

                        # update CVaR state value using state,action value using current alpha in loop (transfering distribution from chosen action)
                        Q_best = Q_CVaR[s,best_action,alpha_i,t]
                        V_CVaR[s,alpha_i,t] = Q_best

                    elif cvar_type=='pCVaR' or cvar_type=='nCVaR':

                        # restrict actions to those allowed in each state (given by task)
                        # not 100% vetted for all types of MDPs
                        actions_allowed = np.array(task.actions_allowed_in_state(s))

                        # otherwise use alpha_i
                        Q_CVaR[s,:,alpha_i,t] = np.round(Q_CVaR[s,:,alpha_i,t],Q_roundoff) # round Q-values so that 'numerical ties' are obvious; not necessary but cleaner for looking at policy
                        Q_best = np.max(Q_CVaR[s,actions_allowed,alpha_i,t])
                        best_actions = np.where(np.squeeze(Q_CVaR[s,:,alpha_i,t])==Q_best)[0]

                        filter=np.isin(best_actions,actions_allowed)
                        best_actions = best_actions[filter]

                        # (implicit tie-breaker to choose the first option)
                        # usually its a single action anyway
                        best_action = best_actions[0]

                        # set policy for current alpha level in loop
                        pi[s,:,alpha_i,t]=0.0
                        pi[s,best_action,alpha_i,t]=1.0

                        # update CVaR state value
                        V_CVaR[s,alpha_i,t] = Q_best

                # evaluate existing policy
                else:
                    # note that this calculation is only valid for deterministic policies
                    V_CVaR[s,alpha_i,t] =  np.sum(policy_to_evaluate[s,:,alpha_i,t]*Q_CVaR[s,:,alpha_i,t])

    output = {}
    output['Q_CVaR']=Q_CVaR
    output['pi']=pi
    output['V_CVaR']=V_CVaR
    output['Xis']=Xis # this can sometimes be large; it's not really necessary for f/n CVaR. So optionally comment out

    return(output)
