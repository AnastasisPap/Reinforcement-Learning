from __future__ import annotations
from libs.envs.grid_world import GridWorldEnv

class DynaMaze(GridWorldEnv):
    def __init__(self, env_args: dict) -> None:
        super().__init__(env_args)
        self.iter_change = env_args.get('iter_change', 10000)
        self.iter = 0
        
        self.obstacles = set(map(tuple, env_args.get('obstacles', [])))
        self.obstacles_after_change = set(map(tuple, env_args.get('obstacles_after_change', [])))
    
    def get_reward(self, s: tuple | int) -> float:
        if self.is_terminal(s): return 1.0
        return 0.0
    
    def step(self, action: int) -> tuple[tuple, float, bool]:
        """Overrides the step function of the GridWorldEnv. The agent moves the same way
        with the difference of having an obstacle. Move as in the GridWorldEnv and if it
        hits an obstacle, then the new state is the same as the previous one.

        Args:
            action (int): the action chosen by the agent 
        Returns:
            State (tuple of ints): the new state after taking the action
            Reward (float): the reward by moving to the new state
            Is terminal (bool): true iff the new state is a terminal one
        """
        self.iter += 1
        if self.iter == self.iter_change:
            self.obstacles = self.obstacles_after_change
        
        prev_s = self._agent_location
        s, _, _ = super().step(action)
        if tuple(s) in self.obstacles: self._agent_location = prev_s

        return self._get_obs(), self.get_reward(self._agent_location), self.is_terminal(self._agent_location)