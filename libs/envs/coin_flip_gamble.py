from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

class CoinFlipGambleEnv(gym.Env):
    def __init__(self, args: dict) -> None:
        self.observation_space = None
        self.max_capital = args.get('max_capital', 100)
        self.p = args.get('p', 0.4)

        self.observation_space = list(range(self.max_capital + 1))
        self.env_dimensions = self.max_capital + 1 # 0 - max_capital
        self.action_space = spaces.Discrete(self.max_capital - 1)
    
    def get_actions(self, s: int) -> list[int]:
        return list(range(min(s, self.max_capital - s) + 1))
    
    def check_state_action(self, s: int, a: int) -> bool:
        return a <= min(s, self.max_capital - s)
    
    def get_dynamics(self):
        dynamics = {}

        for s in self.observation_space:
            for a in self.get_actions(s):
                dynamics[s, a] = {}

                for s_prime in [0] + self.observation_space + [self.max_capital]:
                    if s + a == s_prime:
                        r = 1 if s_prime >= self.max_capital else 0
                        dynamics[s, a][s_prime, r] = self.p
                    if s - a == s_prime:
                        dynamics[s, a][s_prime, 0] = 1 - self.p
        
        dynamics[self.max_capital, 0] = {(self.max_capital, 0): 1}
        return dynamics