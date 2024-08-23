from __future__ import annotations

from libs.envs.grid_world import GridWorldEnv

class CliffWalk(GridWorldEnv):
    def __init__(self, env_args: dict) -> None:
        super().__init__(env_args)
        self.cliff = env_args.get('cliff', [(self.height-1, i) for i in range(1, self.width - 1)])
        self.seed = env_args.get('seed', 0)

    def get_reward(self, state: tuple | int) -> float:
        if tuple(state) in self.cliff: return -100.0
        return -1.0

    def step(self, action: int) -> tuple[tuple, float, bool]:
        s, _, _ = super().step(action)
        r = self.get_reward(self._agent_location)

        if tuple(s) in self.cliff:
            self.reset(self.seed)

        return self._get_obs(), r, self.is_terminal(self._agent_location)