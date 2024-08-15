from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, env_args: dict) -> None:
        self.env_dimensions = env_args.get('dimensions', (6, 9))
        self.height, self.width = self.env_dimensions
        # The grid shows the position of each entity
        # 0: empty, 1: obstacle, 2: agent, 3: goal
        self.grid = np.zeros(self.env_dimensions, dtype=int)
        self.start_state = env_args.get('start_state', (2, 0))
        self.goal_state = env_args.get('goal_state', (0, 8))

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1])
        }
        self._action_to_name = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def _get_obs(self) -> tuple | int:
        return tuple(self._agent_location)
    
    def _get_info(self) -> dict:
        return {"grid": self.grid}
    
    def is_valid_state(self, state: tuple | int) -> bool:
        return np.logical_and(np.greater_equal(state, (0, 0)), np.less(state, self.grid.shape)).all()

    def is_terminal(self, state: tuple | int) -> bool:
        return np.array_equal(state, self.goal_state)

    def get_reward(self, state: tuple | int) -> float:
        return -1.0

    def reset(self, seed: int=0) -> tuple | int:
        """Resets the position of the agent to the start state.

        Args:
            Seed (int): required by the gymnasium interface
        Returns:
            The starting position (tuple of ints) of the agent
        """
        super().reset(seed=seed)
        self._agent_location = np.array(self.start_state)
        self.grid[np.where(self.grid == 2)] = 0

        self.grid[self.start_state] = 2
        self.grid[self.goal_state] = 3
        return self._get_obs()
    
    def edit_grid_values(self, positions: list, values: list) -> None:
        assert len(positions) == len(values), "The number of positions and values must be the same"
        for i in range(len(positions)):
            self.grid[positions[i]] = values[i]
    
    def step(self, action: int) -> tuple[tuple, float, bool]:
        """Step is the main interaction between the agent and the environment.
        The agent chooses an action to perform in the environment, which then
        responds with the next state and the reward from taking the action.

        Args:
            Action (int): an integer in [0, 3], which corresponds to one of the
            available actions.
        Returns:
            New agent state (tuple of ints)
            Reward (float) from moving to the new state
            Is terminal (boolean) which is True iff the new state is a terminal state
        """

        # If the new action is out of bounds, clip it
        state = np.clip(
            self._get_obs() + self._action_to_direction[action],
            (0, 0), (self.height - 1, self.width - 1))

        # Update the grid
        if self.is_valid_state(state):
            self.edit_grid_values([self._get_obs(), tuple(state)], [0, 2])
        self._agent_location = state

        return self._get_obs(), self.get_reward(state), self.is_terminal(state)