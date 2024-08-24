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
        self.c = args.get('c', 2)
        
        self.bandit_type = args.get('bandit_type', 'simple')
        self.step_type = args.get('step_type', 'sample_avg')
    
    def step(self):
        if self.bandit_type == 'ucb':
            # If N is 0, then the action is maximizing
            a = np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))
        else:
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

        self.t += 1

        return a, r