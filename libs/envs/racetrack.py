import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RacetrackEnv(gym.Env):
    def __init__(self, env_args):
        # Racetrack dimensions
        self.env_dimensions = env_args.get('dimensions', (32, 17))
        self.height, self.width = self.env_dimensions

        # used for pygame rendering, to scale the grid
        self.size = env_args.get('size', 20)
        self.render_mode = env_args.get('render_mode', None)
        # Window size of pygame window
        self.window_size = (self.width * self.size, self.height * self.size)
        self.window = None
        self.clock = None
        self._agent_location = None

        # the agent marker is an int which shows the location of the agent in the grid
        self.agent_marker = env_args.get('agent_marker', 4)

        self.action_space = spaces.Discrete(9)
        self._action_to_direction = {
            0: np.array([1, 1]),
            1: np.array([1, -1]),
            2: np.array([1, 0]),
            3: np.array([-1, 1]),
            4: np.array([-1, -1]),
            5: np.array([-1, 0]),
            6: np.array([0, 1]),
            7: np.array([0, -1]),
            8: np.array([0, 0]),
        }

        self.start_states = env_args.get('start_states', [(31,3+i) for i in range(6)])
        self.goal_states = env_args.get('goal_states', [(i,16) for i in range(6)])
        self.boundaries = env_args.get(
            'boundaries',
            [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2)] +\
                [(i+14,0) for i in range(18)]+[(i+22,1) for i in range(10)] +\
                    [(31,2),(30,2),(29,2)]+[(i+7,9) for i in range(25)]+\
                        [(i+6,10+j) for i in range(26) for j in range(7)])

        # Grid[i,j] = 0: empty, 1: wall, 2: start, 3: finish, 4: agent location
        self.grid= np.zeros(self.env_dimensions, dtype=int)
        start_states = tuple(np.array(self.start_states).T)
        goal_states = tuple(np.array(self.goal_states).T)
        boundaries = tuple(np.array(self.boundaries).T)

        self.grid[boundaries] = 1
        self.grid[start_states] = 2
        self.grid[goal_states] = 3

    def reset(self, seed=0):
        super().reset(seed=seed)
        self.vel = np.array([0, 0])

        if self._agent_location is not None and np.less(self._agent_location, [self.height, self.width]).all():
            self.grid[self._agent_location] = 0 if tuple(self._agent_location) not in self.start_states else 2
        self._agent_location = self.start_states[np.random.randint(len(self.start_states))]

        self.grid[self._agent_location] = self.agent_marker

        if self.render_mode == 'human':
            self.render()

        return self._get_obs()

    def _get_obs(self):
        return tuple(self._agent_location)
    
    def is_terminal(self, s):
        rows = np.array(self.goal_states)[:,0]
        col = np.array(self.goal_states)[0,1]

        return s[0] in rows and s[1] >= col

    def out_of_bounds(self, state):
        """Checks whether a state is out of bounds. However, if it is indeed out of bounds,
        but crossed the finish line, then it is not considered out of bounds.
        """
        return (tuple(state) in self.boundaries or\
            np.logical_or(np.less(state, (0, 0)), np.greater_equal(state, self.grid.shape)).any()) and\
            not self.is_terminal(state)
    
    def step(self, action):
        # Increase velocity based on action
        new_vel = self.vel + self._action_to_direction[action]

        # Clip velocity to be at most 3 and at least 0
        self.vel = np.clip(new_vel, 0, 4)
        new_loc = self._agent_location + new_vel * [-1, 1]

        if self.out_of_bounds(new_loc):
            new_loc = self.reset()
        
        new_loc = tuple(new_loc)
        self.grid[self._agent_location] = 0 if tuple(self._agent_location) not in self.start_states else 2
        if np.greater_equal(new_loc, [0,0]).all() and np.less(new_loc, self.grid.shape).all():
            self.grid[new_loc] = self.agent_marker

        self._agent_location = new_loc
        
        if self.render_mode == 'human': self.render()

        return self._get_obs(), -1, self.is_terminal(new_loc)
    
    def render(self):
        if self.window is None:
            pygame.init()
            if self.render_mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))
        fills = {
            0: (160, 160, 160),
            1: (255, 255, 255),
            2: (61, 227, 144),
            3: (235, 52, 52),
            4: (255, 0, 0)
        }

        for i in range(self.height):
            for j in range(self.width):
                # Draw starting and goal states
                if self.grid[i, j] in [2, 3]:
                    pygame.draw.rect(self.window, fills[self.grid[i, j]], (j*self.size, i*self.size, self.size, self.size), 0)

                # Draw boundaries
                pygame.draw.rect(self.window, fills[self.grid[i, j]], (j*self.size, i*self.size, self.size, self.size), 1)

        # Draw agent
        pygame.draw.rect(self.window, (86, 61, 227), (self._agent_location[1]*self.size, self._agent_location[0]*self.size, self.size, self.size))

        if self.render_mode == 'human':
            pygame.display.update()
            for event in pygame.event.get():
                # Check if the user wants to quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.window = None
                    self.clock = None

            self.clock.tick(6) # 6 = 6 frames per second
    