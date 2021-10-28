import numpy as np
import scipy
from scipy.optimize import minimize

def calc_cvar_from_samples(R,alpha):
    '''Calculating CVaR for a distribution represented by samples.

    Args:
        R (np.array): samples from the distribution (these will be reward probabilities [0.4,0.3,0.5,0.4,etc...])
        alpha (float): alpha to calculate CVaR at
    '''
    def _cvar_loss(var,R,alpha):
        Rpos = R-var
        Rpos[Rpos<0]=0
        cvar = var + (1/alpha)*np.mean(Rpos) # trying to minimize that threshold.
        return(cvar)

    var0 = [np.mean(-1*R)]
    bnds = [(np.min(-1*R),np.max(-1*R))]

    results = minimize(_cvar_loss, var0, args=(-1*R,alpha),bounds=bnds,method='SLSQP')

    if results.success:
        cvar =-1*results.fun
        var = -1*results.x
        return(var[0],cvar)
    else:
        return(np.nan, np.nan)

def calc_cvar_distorted_expectation(alpha,P_dist,Z_values):
    '''Calculating CVaR as a distorted expectation, given discrete probability distribution.

    Args:
        P_dist: original discrete probability distribution for Z.
        Z_values: values for the r.v.
           Note: correspond to the probabilities
           Note: This can be return RV, or later V (return conditioned on each state)
           and in that case the density will be over

    Returns:
        weights: Xi-weights, which are being optimized for
        Cvar_est: estimated Cvar
    '''
    assert len(Z_values)==len(P_dist)

    # set objective: maximize (i.e. minimize negative) distorted expectation
    def obj_fun(Xi,Z_values,P_dist):
        return(np.sum((Xi*Z_values*P_dist)))

    # distortion weight bounds
    bnds = tuple(((0,1/alpha) for i in range(len(Z_values))))

    # distortion weight initial values
    Xi_init = np.random.uniform(0,1/alpha,len(Z_values))

    # set equality constraint:
    def sum_to_1_constraint(Xi):
        return(np.dot(Xi,P_dist)-1) # dist is found in env one up
    cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

    results = minimize(obj_fun, Xi_init, args=(Z_values,P_dist), method='SLSQP',
                       bounds=bnds,
                       constraints=cons)

    # unpack results
    Xi_weights = results.x
    Cvar_est = results.fun

    #double check that the constraint aren't broken
    assert np.abs((1-np.dot(Xi_weights,P_dist)))<0.01
    assert np.all(Xi_weights<=(1/alpha+0.01))
    assert np.all(0<=Xi_weights)

    return(Cvar_est, Xi_weights)
