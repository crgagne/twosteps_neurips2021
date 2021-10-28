import numpy as np
from mscl import calc_cvar_from_samples

def run_simulation(task,
                   policy,  # must be states x actions x alphas x time-steps
                   T = 3,
                   Nsims = 10000,
                   s0 = 0,
                   alpha_i0= 0,
                   alpha_set = None,
                   Xis=None,
                   adjust_alpha=False):
    '''Runs a policy in a task (i.e. MDP) multiple times and returns states, actions, rewards.

    Wraps next function.
    '''

    returns = []
    rewards = []
    states = []
    actions  = []
    alphas = []
    for sim in range(Nsims):
        results = run_task(task,T,policy,s0,alpha_i0,alpha_set,Xis,adjust_alpha)
        returns.append(results['R'])
        states.append(results['states'])
        actions.append(results['actions'])
        rewards.append(results['rewards'])
        alphas.append(results['alphas'])

    results = {}
    results['returns']=np.array(returns)
    results['states']=np.array(states)
    results['actions']=np.array(actions)
    results['rewards']=np.array(rewards)
    results['alphas']=np.array(alphas)

    return(results)


def run_task(task,
             T,
             policy,
             s0=0,
             alpha0_i= 0,
             alpha_set = None,
             Xis = None,
             adjust_alpha=False):
    '''Runs a policy in a task (i.e. MDP) and returns states, actions, rewards.
    '''

    s=s0
    alpha_i = alpha0_i
    alpha = alpha_set[alpha_i]
    rewards = []
    states = []
    actions  = []
    alphas = []
    for t in range(T):

        # store state
        states.append(s)
        alphas.append(alpha)

        # choose action
        p_action = policy[s,:,alpha_i,t]
        a = np.random.choice([0,1],p=p_action)
        actions.append(a)

        # get reward for current state
        r = np.random.choice(task.r_support,p=task.p_r[s,:])
        r_i = np.where(task.r_support==r)[0][0]
        rewards.append(r)

        # get next state
        p = task.P[s,:,a]
        if np.sum(p)==0:
            sp = np.nan
        else:
            sp = np.random.choice(np.arange(len(p)),p=p)

        # adjust alpha using Xis if running a risk-dynamic policy
        # this is after action is chosen in state s according to original alpha level
        if adjust_alpha and not np.isnan(sp): # sp will be nan if transition probabilities are all 0
            xi = Xis[s,a,alpha_i,r_i,sp,t]
            alpha = xi*alpha
            alpha_i = np.argmin(np.abs(alpha_set-alpha))
            alpha = alpha_set[alpha_i]

        # set current state to next state
        s = sp

    R=np.sum(rewards)
    results = {}
    results['R']=R
    results['rewards']=rewards
    results['states']=states
    results['actions']=actions
    results['alphas']=alphas
    return(results)

def calc_V_CVaR_MCMC(alpha_set,returns):
    '''Calculates the CVaR of the return distribution for a simulated dataset of trajectories.
    '''

    V_CVaR_MCMC = []
    for alpha in alpha_set:
        if alpha==0.0:
            cvar=np.min(returns)
        else:
            var,cvar = calc_cvar_from_samples(returns,alpha)
        V_CVaR_MCMC.append(cvar)

    return(np.array(V_CVaR_MCMC))

def calc_policy_and_alphas_from_mcmc(results_mcmc):
    '''Calculate the percentages of actions taken in each state during simulation and the adjusted alpha levels.
    '''
    core_states =  [0,1,2,3,4,5,6,7,8,9,10,11]
    Pol_Dyn = np.zeros((len(core_states),2))
    Alpha_Dyn = np.zeros((len(core_states),1))
    Alpha_Dyn_std = np.zeros((len(core_states),1))
    for s in core_states:

        # get the actions that were taken in each state
        actions = results_mcmc['actions'][results_mcmc['states']==s]

        # print the state if the agent took a different action
        if len(np.unique(actions))>1:
            print('s')

        # get the mean probability of action in each case
        # (will be 1 if there is only 1 action taken)
        p = actions.mean()
        Pol_Dyn[s,1]=p
        Pol_Dyn[s,0]=1-p

        # calculated the mean of the alpha values visited in those states
        Alpha_Dyn[s]= results_mcmc['alphas'][results_mcmc['states']==s].mean()
        Alpha_Dyn_std[s]= results_mcmc['alphas'][results_mcmc['states']==s].std()

    return(Pol_Dyn,Alpha_Dyn,Alpha_Dyn_std)
