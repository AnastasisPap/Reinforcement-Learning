import numpy as np

class Bandit:
    def __init__(self, env, args):
        self.epsilon = args.get('epsilon', 0)
        
        self.N = np.zeros(env.n)

        init_value = args.get('init_value', 0.0)
        self.Q = np.full(env.n, init_value)

        self.env = env
        self.t = 0

        self.alpha = args.get('alpha', 0.1)
        
        self.bandit_type = args.get('bandit_type', 'simple')
        self.step_type = args.get('step_type', 'sample_avg')
    
    def step(self):
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(self.env.n)
        else:
            actions = np.where(self.Q == np.max(self.Q))[0]
            a = np.random.choice(actions)
        
        r = self.env.step(a)
        self.N[a] += 1
        if self.step_type == 'sample_avg':
            step_size = 1/self.N[a]
        else: step_size = self.alpha
        self.Q[a] += step_size * (r - self.Q[a])

        return a, r