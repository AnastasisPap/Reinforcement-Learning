from scipy.special import softmax
import numpy as np

class GradientBandit:
    def __init__(self, env, args):
        self.alpha = args.get('alpha', 0.1)
        self.distribution = args.get('distribution', softmax)
        self.baseline = args.get('baseline', True)

        self.H = np.zeros(env.n)
        self.env = env

        self.t = 0
        self.avg_R = 0
        self.N = 0
    
    def step(self):
        probs = self.distribution(self.H)
        a = np.random.choice(self.env.n, p=probs)
        r = self.env.step(a)

        self.N += 1
        self.avg_R += (1/self.N) * (r - self.avg_R)

        baseline = self.avg_R if self.baseline else 0
        self.H[a] += self.alpha * (r-baseline) * (1-probs[a])
        for i in range(self.env.n):
            if i != a:
                self.H[i] -= self.alpha * (r-baseline) * probs[i]

        self.t += 1

        return a, r