import numpy as np

class SimpleBandit:
    def __init__(self, env, args):
        self.epsilon = args.get('epsilon', 0)
        
        self.N = np.zeros(env.n)

        init_value = args.get('init_value', 0.0)
        self.Q = np.full(env.n, init_value)

        self.env = env
        self.t = 0
    
    def step(self, s=None):
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(self.env.n)
        else:
            actions = np.where(self.Q == np.max(self.Q))[0]
            a = np.random.choice(actions)
        
        r = self.env.step(a)
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]

        return a, r