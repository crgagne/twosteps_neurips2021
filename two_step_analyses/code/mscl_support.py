import numpy as np

def state_action_to_index(state,action):
    '''converts states and actions too option index 0-3'''

    if state==1 and action==0:
        return(0)
    elif state==1 and action==1:
        return(1)
    elif state==2 and action==0:
        return(2)
    elif state==2 and action==1:
        return(3)
    else:
        import pdb; pdb.set_trace()

def sigmoid(x):
    return(1.0 / (1.0 + np.exp(-x)))

def inverse_sigmoid(y):
    return(np.log(y/(1-y)))
