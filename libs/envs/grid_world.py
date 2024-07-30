import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(
            self,
            start_state,
            goal_state,
            n_rows=5,
            n_cols=10):
        self.width = n_cols
        self.height = n_rows
        # The grid shows the position of each entity
        # 0: empty, 1: obstacle, 2: agent, 3: goal
        self.grid = np.zeros((n_rows, n_cols), dtype=int)
        self.grid[goal_state] = 3
        self.start_state = start_state
        self.goal_state = goal_state 

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }
        self._action_to_name = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def _get_obs(self):
        return tuple(self._agent_location)
    
    def _get_info(self):
        return {"grid": self.grid}
    
    def is_valid_state(self, state):
        return np.logical_and(np.greater_equal(state, (0, 0)), np.less(state, self.grid.shape)).all()

    def is_terminal(self, state):
        return np.array_equal(state, self.goal_state)

    def get_reward(self, state):
        return -1.0

    """Resets the position of the agent to the start state.

    Args:
        Seed (int): required by the gymnasium interface
    Returns:
        The starting position (tuple of ints) of the agent
    """
    def reset(self, seed=0):
        super().reset(seed=seed)
        self._agent_location = np.array(self.start_state)
        self.grid[np.where(self.grid == 2)] = 0

        self.grid[self.start_state] = 2

        return self.start_state
    
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
    def step(self, action):
        # If the new action is out of bounds, clip it
        state = np.clip(
            self._get_obs() + self._action_to_direction[action],
            (0, 0), (self.height - 1, self.width - 1))

        # Update the grid
        if self.is_valid_state(state):
            self.grid[self._get_obs()] = 0
            self.grid[tuple(state)] = 2
        self._agent_location = state

        return self._get_obs(), self.get_reward(state), self.is_terminal(state)