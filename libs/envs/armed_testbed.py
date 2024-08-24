from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ArmedTestbed(gym.Env):
    def __init__(self, env_args: dict) -> None:
        self.n = env_args.get('n', 10)
        self.mean = env_args.get('mean', 0)
        self.var = env_args.get('var', 1)

    def reset(self, seed: int=0) -> None:
        super().reset(seed=seed)

        self.q_star = np.random.normal(self.mean, self.var, self.n)
    
    def step(self, a: int) -> float:
        return np.random.normal(self.q_star[a], self.var)