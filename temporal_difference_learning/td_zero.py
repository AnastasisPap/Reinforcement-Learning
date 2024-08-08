from libs.utils.agent import BaseAgent

class TDZero(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.V[0] = 0.0
        self.V[-1] = 0.0
    
    def reset(self, s):
        self.s = s
    
    def step(self, s):
        next_s, r, is_term = self.env.step(s)
        self.V[s] += self.alpha * (r + self.gamma * self.V[next_s] - self.V[s])

        return next_s, is_term, r