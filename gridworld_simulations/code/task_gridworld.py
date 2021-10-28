import numpy as np
from scipy.stats import norm
from task_utils import state2idcs

class Task_GridWorld:
    '''Represents the Markov Decision Process (MDP) for which the optimal CVaR-policies will be calculated.

    The key components are the transition probability matrix `P` and the reward distribution `p_r`.
    '''

    def __init__(self):

        # Diagram #
        # [12] quit -3 [0, 1, 2, 3]  [13] goal +10
        #              [4, 5, 6, 7]
        #              [8, 9, 10, 11]
        #                [14] lava -10
        # [15] terminal

        # params
        self.Ns = 16
        self.Na = 2
        self.err_prob_right= 0.08
        self.err_prob_left = 0.04
        self.goal = 13
        self.quit = 12
        self.lava = 14
        self.terminal_state = 15
        self.absorbing_states = [12,13,14,15]
        self.reward_dicts  = {self.quit:-2,self.goal: 3,self.lava:-15}
        self.Rmin = -15
        self.Rmax = 15
        self.dr = 1

        # create empty maze with correct shape for plotting
        self.maze = np.zeros((3,4))

        # build state transition matrix #
        self.P = np.zeros((self.Ns,self.Ns,self.Na))
        for s in range(12):

            # deal with left edge and going left
            if s in [0,4,8]:
                self.P[s,self.quit,0] = 1-self.err_prob_left # go left and quit
            else:
                self.P[s,s-1,0] = 1-self.err_prob_left # go left

            # deal with right edge and going right
            if s in [3,7,11]:
                self.P[s,self.goal,1] = 1-self.err_prob_right # go right to goal
            else:
                self.P[s,s+1,1] = 1-self.err_prob_right # go right

            # deal with bottom row and falling off
            if s in [8,9,10,11]:
                self.P[s,self.lava,0] = self.err_prob_left # go left but down
                self.P[s,self.lava,1] = self.err_prob_right # go right but down
            else:
                self.P[s,s+4,0] = self.err_prob_left # go left but down
                self.P[s,s+4,1] = self.err_prob_right # go right but down

        # states 8,9,10 transition to terminal state with either action
        for s in self.absorbing_states:
            self.P[s,self.terminal_state,:]=1

        # reward range and possible values
        self.r_support = np.arange(self.Rmin,self.Rmax+self.dr,self.dr) # [-1,0,1,2,3]
        self.Nr = len(self.r_support)

        # create reward distributions per state (or deterministc in this case),
        # and a list of rewards associated with each state
        self.p_r = np.zeros((self.Ns,self.Nr))
        self.rewards = np.zeros(self.Ns)

        for state in range(self.Ns):
            if state in self.reward_dicts.keys():
                r =self.reward_dicts[state]
            else:
                r = 0 # fill in all non-rewarded states with reward = 0
            self.rewards[state]=r
            self.p_r[state,np.where(self.r_support==r)]=1.0 # deterministic rewards

    def states_allowed_at_time(self,t):
        return([s for s in range(self.Ns)])

    def actions_allowed_in_state(self,s):
        allowed = [0,1]
        return(allowed)
