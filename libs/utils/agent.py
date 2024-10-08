from __future__ import annotations
import gymnasium as gym
import numpy as np

class BaseAgent:
    def __init__(self, env: gym.Env, args: dict) -> None:
        """
        Args:
            Env: object of DynaMaze which is the agent environment
            args: dictionary mapping name (str) -> value (int or float)
        """
        self.epsilon = args.get('epsilon', 0.1)
        self.gamma = args.get('gamma', 1.0)
        self.alpha = args.get('alpha', 0.1)
        self.env = env
        self.rnd_gen = np.random
        self.policy = None

        init_value = args.get('init_value', 0.0)
        dtype = args.get('dtype', float)
        dims = (env.env_dimensions,) if type(env.env_dimensions) is int else env.env_dimensions

        if init_value == 'uniform':
            self.Q = np.random.uniform(size=(*dims, env.action_space.n))
            self.V = np.random.uniform(size=dims)
        elif init_value == 'normal':
            self.Q = np.random.normal(size=(*dims, env.action_space.n))
            self.V = np.random.normal(size=dims)
        else:
            self.Q = np.full((*dims, env.action_space.n), init_value, dtype=dtype)
            self.V = np.full(dims, init_value, dtype=dtype)
    
    def eps_greedy(self, s: tuple | int) -> int:
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
    
    def reset(self, args: dict):
        return
        
    def step(self, s: tuple | int) -> tuple:
        return None, None, None