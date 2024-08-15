import numpy as np

from libs.envs.grid_world import GridWorldEnv

class WindyGridWorld(GridWorldEnv):
    def __init__(self, env_args: dict) -> None:
        super().__init__(env_args)
        self.wind_strength = env_args.get('wind_strength', [0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    
    def step(self, action: int) -> tuple[tuple, float, bool]:
        wind = self.wind_strength[self._agent_location[1]]
        s, _, _ = super().step(action)

        self._agent_location = np.clip(
            s - np.array([wind, 0]),
            (0, 0), (self.height - 1, self.width - 1))
        
        return self._get_obs(), self.get_reward(self._agent_location), self.is_terminal(self._agent_location)