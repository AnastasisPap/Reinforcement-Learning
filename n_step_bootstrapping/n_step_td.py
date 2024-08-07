import numpy as np
import matplotlib.pyplot as plt

from libs.utils.agent import BaseAgent
from libs.envs.random_walk import RandomWalk
from libs.utils.experiment import experiment

class NStepTD(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.n = args.get('n', 1)
    
    def reset(self, s):
        self.states = [s]
        self.rewards = [0]
        self.t = 0
        self.T = float('inf')
    
    def step(self, s):
        finished = False

        if self.t < self.T:
            s, r, is_term = self.env.step(s)
            self.states.append(s)
            self.rewards.append(r)

            if is_term: self.T = self.t + 1
        
        tau = self.t - self.n + 1
        if tau >= 0:
            G = 0.0
            for i in range(tau+1, min(tau+self.n, self.T)+1):
                G += pow(self.gamma, i-tau-1) * self.rewards[i]
            
            if tau + self.n < self.T:
                G += pow(self.gamma, self.n) * self.V[self.states[tau+self.n]]

            if not self.env.is_terminal(self.states[tau]):
                self.V[self.states[tau]] += self.alpha * (G - self.V[self.states[tau]])
        
        if tau == self.T - 1:
            finished = True
        self.t += 1

        return s, finished, 0
    