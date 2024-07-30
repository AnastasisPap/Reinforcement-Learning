import sys, os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from libs.envs.grid_world import GridWorldEnv
import numpy as np

class DynaMaze(GridWorldEnv):
    def __init__(self, start_state, goal_state, n_rows=6, n_cols=9):
        super().__init__(start_state, goal_state, n_rows, n_cols)
        self.obstacles = []
    
    def get_reward(self, s: tuple[int, int] | None) -> float:
        if self.is_terminal(s): return 1.0
        return 0.0

    """Adds a wall (obstacle) to the environment. If an agent moves towards an obstacle,
    it stays in the same position. From start pos until end pos, the grid will have value of 1.
    It's required that start pos and end pos either have the same row (move horizontally) or
    column (move vertically). Also both must be valid positions.

    Args:
        Start pos (tuple): the starting position of the wall
        End pos (tuple): the last position of the wall

    """
    def add_wall(self, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> None:
        assert start_pos[0] == end_pos[0] or start_pos[1] == end_pos[1], "Start pos must have the same column or row with end pos."

        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        step = 1 - (start_pos == end_pos) # Determines whether to move horizontally or vertically
        direction = np.sign(end_pos - start_pos) # Determines the direction to move (e.g. move up/down or left/right)
        step *= direction

        curr_pos = start_pos

        # Add each state to the obstacles list and update the grid
        self.obstacles.append(tuple(curr_pos))
        if self.is_valid_state(curr_pos):
            self.grid[tuple(curr_pos)] = 1
        while not np.array_equal(curr_pos, end_pos):
            curr_pos += step
            self.obstacles.append(tuple(curr_pos))
            
            if self.is_valid_state(curr_pos):
                self.grid[tuple(curr_pos)] = 1

    """Overrides the step function of the GridWorldEnv. The agent moves the same way
    with the difference of having an obstacle. Move as in the GridWorldEnv and if it
    hits an obstacle, then the new state is the same as the previous one.
    """
    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        state = self._get_obs()
        next_state, _, _ = super().step(action)

        if next_state in self.obstacles:
            self._agent_location = np.array(state)
            self.grid[next_state] = 1
            self.grid[state] = 2

        return self._get_obs(), self.get_reward(next_state), self.is_terminal(next_state)