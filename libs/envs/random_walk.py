import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RandomWalk(gym.Env):
    def __init__(self, env_args: dict) -> None:
        self.n = env_args.get('n', 5)

        self.action_space = spaces.Discrete(2)
        self.start_state = self.n // 2 + 1
        self.left_r = env_args.get('left_r', 0)
        self.right_r = env_args.get('right_r', 1)
        self.env_dimensions = self.n + 2
    
    def reset(self, seed: int=0) -> int:
        super().reset(seed=seed)
        self._agent_location = self.start_state

        return self._agent_location
    
    def _get_obs(self) -> int:
        return self._agent_location
    
    def is_terminal(self, s: int) -> bool:
        return s == 0 or s == self.n + 1
    
    def get_reward(self, s: int) -> int:
        if s == 0: return self.left_r
        elif s == self.n + 1: return self.right_r
        return 0

    # The action is included as an argument for the sake of consistency
    def step(self, a: int) -> tuple[int, int, bool]:
        self._agent_location += np.random.choice([-1, 1])

        return self._get_obs(), self.get_reward(self._agent_location), self.is_terminal(self._agent_location)