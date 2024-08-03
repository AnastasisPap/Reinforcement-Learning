import sys, os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from libs.envs.grid_world import GridWorldEnv
import numpy as np

class DynaMaze(GridWorldEnv):
    def __init__(self, env_args):
        super().__init__(env_args)
        self.iter_change = env_args.get('iter_change', 10000)
        self.iter = 0
        
        self.obstacles = set(map(tuple, env_args.get('obstacles', [])))
        self.obstacles_after_change = set(map(tuple, env_args.get('obstacles_after_change', [])))
    
    def get_reward(self, s):
        if self.is_terminal(s): return 1.0
        return 0.0
    
    def step(self, action):
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