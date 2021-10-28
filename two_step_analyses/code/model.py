import numpy as np
from scipy.optimize import minimize
from mscl_support import state_action_to_index, sigmoid, inverse_sigmoid
from scipy.stats import beta as Beta
from scipy.stats import norm
import scipy
import multiprocessing
from multiprocessing import Pool

# Some constants for parameter constraints
asymp_var = 0.1
lr_min=0.01;lr_max=0.99
eta_min=0.001;eta_max=asymp_var-0.01
alpha_min=0.1;alpha_max=1.0
tau_max = 30

def untransform_params(param,param_name):
    '''Converts unbounded parameter values into bounded ones
        e.g. R-->[0,1] for learning rate
    '''
    if param_name=='lambda':
        param = (lr_max-lr_min)*sigmoid(param)+lr_min
    elif 'tau' in param_name:
        param=np.exp(param)
        if param>tau_max: # extra limit on tau to prevent overflow
            param=tau_max
    elif param_name=='eta':
        param=(eta_max-eta_min)*sigmoid(param)+eta_min
    elif 'cvar_alpha'==param_name:
        param=(alpha_max-alpha_min)*sigmoid(param)+alpha_min
    return(param)

def transform_params(param,param_name):
    '''Inverts the above transformation for each parameter'''
    if param_name=='lambda':
        param = inverse_sigmoid((param-lr_min)/(lr_max-lr_min))
    elif 'tau' in param_name:
        param=np.log(param)
    elif param_name=='eta':
        param=inverse_sigmoid((param-eta_min)/(eta_max-eta_min))
    elif 'cvar_alpha'==param_name:
        param=inverse_sigmoid((param-alpha_min)/(alpha_max-alpha_min))
    return(param)

def model(params,
          model_name,
          stage1_choices=None,
          stage2_choices=None,
          stage2_states=None,
          outcomes=None,
          return_nllk=True,
          transform=True,
          verbose=0,
          generative=False, # if using the model to generate data
          drifting_probs=None,
          single_state=False,
          T=200,
          use_shortcut_for_mean=True, # skips cvar calculations for mean model
          seed_per_trial=False, # reset random seed per trial, not for fitting
          forced_stage2_choices = None,
          forced_stage2_outcomes= None,
          force_choices_until=0,
          force_outcomes_until=0,
          deterministic_2nd_choice=False,
          init_Psi2=0.01):

    '''Main model function for both the CVaR- and the mean-model.
       This can be called directly (e.g. for generating data from model),
       but it is also used in fit_model() below.
    '''

    # choose model
    if model_name=='mean_mb_mf_sticky': # main mean-model
        param_names = ['lambda', 'tau_stage2', 'tau_MB','tau_MF1','tau_sticky']
        if transform:
            params = [untransform_params(param,param_name) for param,param_name in zip(params,param_names)]
        lambdaa=params[0]
        beta_stage2=params[1]
        beta_MB=params[2]
        beta_MF0=0.0
        beta_MF1=params[3]
        beta_sticky=params[4]
        eta2=0.1
        cvar_alpha=1.0
        restrict_lambda = False
        uppertail=False
    elif model_name=='dcvar_mb_mf_sticky': # main cvar-model
        param_names = ['lambda', 'tau_stage2', 'tau_MB','tau_MF1','tau_sticky','eta','cvar_alpha']
        if transform:
            params = [untransform_params(param,param_name) for param,param_name in zip(params,param_names)]
        lambdaa=params[0]
        beta_stage2=params[1]
        beta_MB=params[2]
        beta_MF0=0.0
        beta_MF1=params[3]
        beta_sticky=params[4]
        eta2=params[5]
        cvar_alpha=params[6]
        restrict_lambda = True
        uppertail=False
    elif model_name=='dcvar_mb_mf': # cvar-model without perseveration
        param_names = ['alpha', 'beta_stage2', 'beta_MB','beta_MF1','eta','cvar_alpha']
        if transform:
            params = [untransform_params(param,param_name) for param,param_name in zip(params,param_names)]
        lambdaa=params[0]
        beta_stage2=params[1]
        beta_MB=params[2]
        beta_MF0=0.0
        beta_MF1=params[3]
        beta_sticky=0
        eta2=params[4]
        cvar_alpha=params[5]
        restrict_lambda = True
        beta_sticky_stage2 = 0.0
        uppertail=False

    # for debugging
    if verbose>0:
        print(params)

    # extra catches for parameter ranges
    if cvar_alpha>1.0:
        cvar_alpha=1.0
    if eta2<0.001:
        eta2=0.001

    # given eta and asymptotic variance, calculate phi
    phi = 1-np.sqrt(1-(eta2/asymp_var))

    sampled_drifting_probs = []

    # set up data storage if not passed in (for genertive)
    if stage1_choices is None:
        stage1_choices = []
        stage2_choices = []
        stage2_states = []
        outcomes = []
    else:
        T = len(stage1_choices)

    # second 2 stuff
    Q_stage2 = np.ones((T,2,2))*0.5 # two possible states, two possible actions
    Psi2_stage2 = np.ones((T,2,2))*init_Psi2
    Q_stage2_CVaR = np.ones((T,2,2))*0.5
    p_stage2_choices = np.zeros((T,2,2))

    # calculate CVaR for initial trial
    Q_stage2_CVaR[0,:,:] = Q_stage2[0,:,:] - (1.0/(cvar_alpha))*norm.pdf(norm.ppf(1-cvar_alpha))*np.sqrt(Psi2_stage2[0,:,:])

    # stage 1 model-free TD-0
    Q_MF0 = np.ones((T,2))*0.5
    Psi2_MF0 = np.ones((T,2))*init_Psi2
    Q_MF0_CVaR = np.ones((T,2))*0.5

    # stage 1 model-free TD-1
    Q_MF1 = np.ones((T,2))*0.5
    Psi2_MF1 = np.ones((T,2))*init_Psi2
    Q_MF1_CVaR = np.ones((T,2))*0.5

    # stage 1 model-based
    Q_MB = np.zeros((T,2))
    Q_MB_CVaR = np.zeros((T,2))

    p_stage1_choices = np.zeros((T,2))
    choice_probs = np.zeros((T,3,2)) # will be both stage 1 and stage 2

    # stage 1 --> stage 2 probabilies
    P = np.array([[0.7,0.3],
                  [0.3,0.7]]) # transitions from each top action to lower state

    # penalize broken constraints; can help find good parameter regime
    if restrict_lambda:
        if lambdaa>((1-phi)**2):
            llk=-100
        else:
            llk=0
    else:
        llk=0

    llk_stage1 = 0.0
    llk_stage2 = 0.0

    # loop through trials
    for t in range(T):

        # if fitting to subjects with missing data
        missing_response1 = False
        missing_response2 = False

        # stage 1 calculations

        # figure out what action the model would take in stage 2 (before arriving there)
        second_stage_choice_state0=np.argmax(Q_stage2_CVaR[t,0,:])
        second_stage_choice_state1=np.argmax(Q_stage2_CVaR[t,1,:])

        # calculate stage 1 model-based distributions and CVaR
        if 'dcvar' in model_name or not use_shortcut_for_mean:

            # get the means and variances for the two bottom stage options that you will choose
            mu1 = Q_stage2[t,0,second_stage_choice_state0]
            mu2 = Q_stage2[t,1,second_stage_choice_state1]
            psi1 = Psi2_stage2[t,0,second_stage_choice_state0]
            psi2 = Psi2_stage2[t,1,second_stage_choice_state1]

            # create p densities for them
            res = 2
            dp = 1.0*10**(-res)
            ps = np.arange(-2,2,dp)
            density1 = norm(loc=mu1,scale=np.sqrt(psi1)).pdf(ps)
            density2 = norm(loc=mu2,scale=np.sqrt(psi2)).pdf(ps)

            # create two mixtures densities
            mixture_density0 = P[0,0]*density1+P[0,1]*density2 # option 0: 70% comes from state 1 and 30% from state 2
            mixture_density1 = P[1,0]*density1+P[1,1]*density2 # option 0: 70% comes from state 1 and 30% from state 2
            mixture_density0_normed = mixture_density0/np.sum(mixture_density0)
            mixture_density1_normed = mixture_density1/np.sum(mixture_density1)

            # calculate VaR
            var0 = np.max(ps[np.round(np.cumsum(mixture_density0_normed),res)<=cvar_alpha]) # rounding helps with alpha=1.0 so it doesn't fluctuate between being equal to an alpha level and not
            var1 = np.max(ps[np.round(np.cumsum(mixture_density1_normed),res)<=cvar_alpha])

            # calculate CVaR
            Q_MB_CVaR[t,0] = dp*(1.0/cvar_alpha)*np.sum(mixture_density0[ps<=var0]*ps[ps<=var0])
            Q_MB_CVaR[t,1] = dp*(1.0/cvar_alpha)*np.sum(mixture_density1[ps<=var1]*ps[ps<=var1])

        # or mean
        elif 'mean' in model_name and use_shortcut_for_mean:

            Q_MB_CVaR[t,0] = P[0,0]*Q_stage2[t,0,second_stage_choice_state0]+P[0,1]*Q_stage2[t,1,second_stage_choice_state1]
            Q_MB_CVaR[t,1] = P[1,0]*Q_stage2[t,0,second_stage_choice_state0]+P[1,1]*Q_stage2[t,1,second_stage_choice_state1]

        if seed_per_trial:
            np.random.seed(t)

        # get indicator [0,1] or [1,0] for last choice
        if t>0:
            sticky_indicator = np.array([(stage1_choices[t-1]==0).astype('float'),(stage1_choices[t-1]==1).astype('float')])
        else:
            sticky_indicator = np.array([0.0,0.0])

        # combined choice value stage 1
        exp_choice_value_stage1 = np.exp(beta_MB*Q_MB_CVaR[t,:] +\
                                         beta_MF1*Q_MF1_CVaR[t,:] +\
                                         beta_MF0*Q_MF0_CVaR[t,:] +\
                                         beta_sticky*sticky_indicator)

        # choice probability stage 1
        p_stage1_choice = exp_choice_value_stage1 / np.sum(exp_choice_value_stage1)
        p_stage1_choices[t,:]=p_stage1_choice

        # choice stage 1 (either generate or get participant's choice)
        if generative:
            stage1_choice = np.random.choice([0,1],p=p_stage1_choices[t,:])
            stage1_choices.append(stage1_choice)
        else:
            stage1_choice = stage1_choices[t]
            if stage1_choice==-1:
                missing_response1=True

        # update likelihood
        if missing_response1:
            pass
        else:
            p_stage1_choice_chosen = p_stage1_choice[stage1_choice]
            llk += np.log(p_stage1_choice_chosen)
            llk_stage1 += np.log(p_stage1_choice_chosen)

        # get transition from stage 1 to stage 2
        if generative:
            stage2_state = np.random.choice([1,2],p=P[stage1_choice,:]) # use indices
            if single_state: # for debugging and for generate and recover on a single lower state
                stage2_state = 1
            stage2_state_idx = stage2_state -1
            stage2_states.append(stage2_state)
        else:
            stage2_state = stage2_states[t]
            stage2_state_idx  = stage2_state - 1 # because it's 2,3 and I want to use as indicators

        # combined choice value stage 2
        exp_choice_value_stage2 = np.exp(beta_stage2*Q_stage2_CVaR[t,:,:])

        # choice probability stage 2
        p_stage2_choice =  exp_choice_value_stage2 / np.sum(exp_choice_value_stage2,axis=1).reshape(-1,1)
        p_stage2_choices[t,:,:]=p_stage2_choice

        # store choice probabilities for plotting
        choice_probs[t,0,:]=p_stage1_choice
        choice_probs[t,1::,:]=p_stage2_choice

        # choice stage 2 (generative or participant's)
        if generative:
            if (forced_stage2_choices is not None) and (t<force_choices_until):
                stage2_choice = forced_stage2_choices[t]
            else:
                p_tmp = p_stage2_choices[t,stage2_state_idx,:]
                if deterministic_2nd_choice:
                    stage2_choice = np.argmax(p_tmp)
                else:
                    stage2_choice =  np.random.choice([0,1],p=p_tmp)
            stage2_choices.append(stage2_choice)
        else:
            stage2_choice = stage2_choices[t]
            if stage2_choice==-1:
                missing_response2=True

        # update likelihood
        if missing_response1 or missing_response2:
            pass
        else:
            p_stage2_choice_chosen = p_stage2_choice[stage2_state_idx,stage2_choice]
            llk += np.log(p_stage2_choice_chosen)
            llk_stage2+= np.log(p_stage2_choice_chosen)

        # get outcome (reward)
        if generative:
            if (forced_stage2_outcomes is not None) and (t<force_outcomes_until):
                outcome = forced_stage2_outcomes[stage2_state_idx,stage2_choice,t]
                drift_prob = np.nan
            else:
                drift_prob = drifting_probs[t,state_action_to_index(stage2_state,stage2_choice)]
                outcome = np.random.choice([0,1],p=[1-drift_prob,drift_prob])
            outcomes.append(outcome)
            sampled_drifting_probs.append(drift_prob)
        else:
            outcome = outcomes[t]

        if missing_response1 or missing_response2:
            assert outcome == -1

        # updating distributional quantities (i.e. learning)
        if t<(T-1) and (not missing_response1) and (not missing_response2):

            # update stage 2 means
            Q_stage2[t+1,stage2_state_idx,stage2_choice]   = Q_stage2[t,stage2_state_idx,stage2_choice] + lambdaa*(outcome - Q_stage2[t,stage2_state_idx,stage2_choice]) # chosen
            Q_stage2[t+1,stage2_state_idx,1-stage2_choice] = Q_stage2[t,stage2_state_idx,1-stage2_choice] + lambdaa*(0.5 - Q_stage2[t,stage2_state_idx,1-stage2_choice]) # same state other choice
            Q_stage2[t+1,1-stage2_state_idx,:]             = Q_stage2[t,1-stage2_state_idx,:] + lambdaa*(0.5 - Q_stage2[t,1-stage2_state_idx,:]) # other state, other two choices

            # update stage 2 variances
            Psi2_stage2[t+1,stage2_state_idx,stage2_choice]   = ((1-phi)**2)*Psi2_stage2[t,stage2_state_idx,stage2_choice] + eta2 - lambdaa*Psi2_stage2[t,stage2_state_idx,stage2_choice]
            Psi2_stage2[t+1,stage2_state_idx,1-stage2_choice] = ((1-phi)**2)*Psi2_stage2[t,stage2_state_idx,1-stage2_choice] + eta2
            Psi2_stage2[t+1,1-stage2_state_idx,:]             = ((1-phi)**2)*Psi2_stage2[t,1-stage2_state_idx,:] + eta2

            # calculate stage 2 CVaRs
            if uppertail:
                Q_stage2_CVaR[t+1,:,:] = Q_stage2[t+1,:,:] + (1.0/(cvar_alpha))*norm.pdf(norm.ppf(cvar_alpha))*np.sqrt(Psi2_stage2[t+1,:,:])
            else:
                Q_stage2_CVaR[t+1,:,:] = Q_stage2[t+1,:,:] - (1.0/(cvar_alpha))*norm.pdf(norm.ppf(1-cvar_alpha))*np.sqrt(Psi2_stage2[t+1,:,:])

            # update stage 1 means (using outcome)
            Q_MF1[t+1,stage1_choice]   = Q_MF1[t,stage1_choice] + lambdaa*(outcome-Q_MF1[t,stage1_choice])
            Q_MF1[t+1,1-stage1_choice] = Q_MF1[t,1-stage1_choice] + lambdaa*(0.5 - Q_MF1[t,1-stage1_choice])

            # update stage 1 variances (using outcome)
            Psi2_MF1[t+1,stage1_choice]   = ((1-phi)**2)*Psi2_MF1[t,stage1_choice] + eta2 - lambdaa*Psi2_MF1[t,stage1_choice]
            Psi2_MF1[t+1,1-stage1_choice] = ((1-phi)**2)*Psi2_MF1[t,1-stage1_choice] + eta2

            # calculate stage 1 CVaRs (using outcome)
            if uppertail:
                Q_MF1_CVaR[t+1,:] = Q_MF1[t+1,:] + (1.0/(cvar_alpha))*norm.pdf(norm.ppf(cvar_alpha))*np.sqrt(Psi2_MF1[t+1,:])
            else:
                Q_MF1_CVaR[t+1,:] = Q_MF1[t+1,:] - (1.0/(cvar_alpha))*norm.pdf(norm.ppf(1-cvar_alpha))*np.sqrt(Psi2_MF1[t+1,:])

            # update stage 1 means (using stage 2 Q-values)
            Q_MF0[t+1,stage1_choice]   = Q_MF0[t,stage1_choice] + lambdaa*(Q_stage2[t,stage2_state_idx,stage2_choice] -  Q_MF0[t,stage1_choice])
            Q_MF0[t+1,1-stage1_choice] = Q_MF0[t,1-stage1_choice] + lambdaa*(0.5 - Q_MF0[t,1-stage1_choice])

            # update stage 1 variances (using stage 2 Q-values)
            Psi2_MF0[t+1,stage1_choice]   = ((1-phi)**2)*Psi2_MF0[t,stage1_choice] + eta2 - lambdaa*Psi2_MF0[t,stage1_choice]
            Psi2_MF0[t+1,1-stage1_choice] = ((1-phi)**2)*Psi2_MF0[t,1-stage1_choice] + eta2

            # calculate stage 1 CVaRs (using stage 2 Q-values)
            if uppertail:
                Q_MF0_CVaR[t+1,:] = Q_MF0[t+1,:] + (1.0/(cvar_alpha))*norm.pdf(norm.ppf(cvar_alpha))*np.sqrt(Psi2_MF0[t+1,:])
            else:
                Q_MF0_CVaR[t+1,:] = Q_MF0[t+1,:] - (1.0/(cvar_alpha))*norm.pdf(norm.ppf(1-cvar_alpha))*np.sqrt(Psi2_MF0[t+1,:])


        elif missing_response1 or missing_response2:

            Q_stage2[t+1,:,:]   = Q_stage2[t,:,:]
            Psi2_stage2[t+1,:,:]   = Psi2_stage2[t,:,:]
            Q_stage2_CVaR[t+1,:,:] = Q_stage2_CVaR[t,:,:]

            Q_MF1[t+1,:]   = Q_MF1[t,:]
            Psi2_MF1[t+1,:]   = Psi2_MF1[t,:]
            Q_MF1_CVaR[t+1,:] = Q_MF1_CVaR[t,:]

            Q_MF0[t+1,:]   = Q_MF0[t,:]
            Psi2_MF0[t+1,:]   = Psi2_MF0[t,:]
            Q_MF0_CVaR[t+1,:] = Q_MF0_CVaR[t+1,:]

    # flip liklihood for minimization
    nllk = -1.0*llk
    nllk_stage1 = -1.0*llk_stage1
    nllk_stage2 = -1.0*llk_stage2

    if verbose>0:
        print(nllk)
        print()

    # when fitting model, return the nllk
    if return_nllk:
        return(nllk)

    # otherwise, return all the internal variables for plotting
    else:
        results = {}
        results['Q_stage2'] = Q_stage2
        results['Psi2_stage2'] = Psi2_stage2
        results['Q_stage2_CVaR'] = Q_stage2_CVaR
        results['Q_MF0'] = Q_MF0
        results['Psi2_MF0'] = Psi2_MF0
        results['Q_MF0_CVaR'] = Q_MF0_CVaR
        results['Q_MF1'] = Q_MF1
        results['Psi2_MF1'] = Psi2_MF1
        results['Q_MF1_CVaR'] = Q_MF1_CVaR
        results['Q_MB'] = Q_MB
        results['Q_MB_CVaR'] = Q_MB_CVaR
        results['p_stage1_choices']=p_stage1_choices
        results['p_stage2_choices']=p_stage2_choices
        results['choice_probs']=choice_probs
        results['params']=params
        results['nllk']=nllk
        results['nllk_stage1']=nllk_stage1
        results['nllk_stage2']=nllk_stage2
        results['stage1_choices']=stage1_choices
        results['stage2_choices']=stage2_choices
        results['stage2_states']=stage2_states
        results['outcomes']=outcomes
        results['sampled_drifting_probs']=sampled_drifting_probs
        results['param_names']=param_names
        results['model_name']=model_name
        results['phi']=phi
        return(results)

def minimize_helper(params_init,args,method,maxiter,tol,bnds=None):
    '''Convenient wrapper for scipy's minimize function'''
    if method is None:
        fit_results = minimize(model,params_init,args=args)
        return(fit_results)
    if bnds is not None:
        fit_results = minimize(model,params_init,args=args,method=method,tol=tol,bounds=bnds)
        return(fit_results)
    if maxiter is None:
        fit_results = minimize(model,params_init,args=args,method=method,tol=tol)
        return(fit_results)
    else:
        fit_results = minimize(model,params_init,args=args,method=method,tol=tol,options={'maxiter':maxiter})
        return(fit_results)

def sample_param(param_name):
    '''Samples parameters from a more restricted range for initial values for optimization'''

    if param_name=='lambda':
        return(np.random.uniform(0.15,0.75))
    elif 'tau_sticky' in param_name:
        return(np.random.uniform(2,10))
    elif 'tau' in param_name:
        return(np.random.uniform(2,10))
    elif param_name=='eta':
        return(np.random.uniform(0.001,asymp_var-0.01))
    elif 'cvar_alpha' in param_name:
        return(np.random.uniform(0.3,0.99))

def fit_model(data,
              model_name,
              multi_starts=10,
              parrallel=True,
              tol=1e-4,
              method='L-BFGS-B',
              transform=True,
              verbose=0):

    '''This is main function used to fit the model to an individual participant's dataset'''

    if method=='L-BFGS-B':
        maxiter=None
    elif method=='nelder-mead':
        tol = 0.1
        maxiter=150
    print(method)

    # choose model
    if model_name=='mean_mb_mf_sticky':
        param_names = ['lambda', 'tau_stage2', 'tau_MB','tau_MF1','tau_sticky']
    elif model_name=='dcvar_mb_mf_sticky':
        param_names = ['lambda', 'tau_stage2', 'tau_MB','tau_MF1','tau_sticky','eta','cvar_alpha']
        bnds = ((0.01,0.99),(0.1,30),(0.1,30),(0.1,30),(0.1,20),(0.001,asymp_var-0.01),(0.1,0.99))
    elif model_name=='mean_mb_mf2_sticky':
        param_names = ['lambda', 'tau_stage2', 'tau_MB','tau_MF0','tau_MF1','tau_sticky']

    # set up inputs for parrallel fitting
    inputs = []
    for _ in range(multi_starts):
        params_init = [sample_param(param_name) for param_name in param_names]
        if transform:
            params_init_t = [transform_params(param,param_name) for param,param_name in zip(params_init,param_names)]
            bnds = None
        else:
            params_init_t = params_init
            bnds = bnds
        args = (model_name,data['stage1_choices'],data['stage2_choices'], data['stage2_states'],data['outcomes'],True,transform,verbose)
        inputs.append((params_init_t,args,method,maxiter,tol,bnds))

    # run in parrallel
    if parrallel:
        with Pool(multiprocessing.cpu_count()) as p:
            map_result = p.starmap(minimize_helper, inputs)
    else:
        map_result = []
        for i in range(multi_starts):
            map_result.append(minimize_helper(inputs[i][0],inputs[i][1],inputs[i][2],inputs[i][3],inputs[i][4],inputs[i][5]))

    # unpack results
    if method=='L-BFGS-B':
        best_nllk = np.inf
        fit_results_list =[]
        for i in range(multi_starts):
            fit_results = map_result[i]
            fit_results_list.append(fit_results)
            if fit_results.fun<best_nllk and fit_results.success:
                best_fit_results = fit_results
                best_nllk = fit_results.fun

    # run the model again with the best fit parameters to get internal variables for plotting
    fit_params = best_fit_results.x
    results = model(fit_params,
               model_name,
               data['stage1_choices'],
               data['stage2_choices'],
               data['stage2_states'],
               data['outcomes'],
               return_nllk=False,
               transform=transform,
               verbose=verbose)
    results['fit_params']=fit_params
    results['nllk']= best_fit_results.fun
    results['fit_results']=best_fit_results
    results['map_result']=map_result

    # get gradients and hessian at minimum (if using L-BFGS-B)
    results['jac']=best_fit_results.jac
    results['H']=np.linalg.inv(best_fit_results.hess_inv.todense())

    # return rest of initial starts
    results['fit_results_list']=fit_results_list

    return(results)
