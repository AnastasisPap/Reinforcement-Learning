import numpy as np

from libs.utils.agent import BaseAgent

class Sarsa(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rnd_gen = np.random.RandomState(args.get('seed', 0))
    
    def reset(self, s):
        self.a = self.eps_greedy(s) 

    def step(self, s):
        next_s, r, is_terminal = self.env.step(self.a)
        next_a = self.eps_greedy(next_s)

        self.Q[s][self.a] += self.alpha * (r + self.gamma * self.Q[next_s][next_a] - self.Q[s][self.a])
        self.a = next_a

        return next_s, is_terminal, r