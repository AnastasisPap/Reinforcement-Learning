import numpy as np

class BaseAgent:
    def __init__(self, env, args):
        """
        Args:
            Env: object of DynaMaze which is the agent environment
            args: dictionary mapping name (str) -> value (int or float)
        """
        self.epsilon = args.get('epsilon', 0.1)
        self.gamma = args.get('gamma', 1.0)
        self.alpha = args.get('alpha', 0.1)
        self.env = env

        self.Q = np.zeros((env.height, env.width, env.action_space.n))
    
    def eps_greedy(self, s):
        """It's the epsilon greedy policy. With probability epsilon selects any action
        randomly with the same probability, and with prob 1-epsilon, greedily selects
        the action and breaks ties arbitrarily.

        Args:
            The state (tuple of ints) for which the policy will be used.
        Returns:
            The action (int) which is an int in [0, 3]) which the policy chose.
        """
        prob = self.rnd_gen.uniform()
        if prob < self.epsilon:
            return self.rnd_gen.randint(self.env.action_space.n)
        else:
            return self.rnd_gen.choice(np.where(np.max(self.Q[s]) == self.Q[s])[0])
        
    def step(self, s):
        return None, None, None