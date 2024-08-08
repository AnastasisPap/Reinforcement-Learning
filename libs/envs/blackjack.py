import gymnasium as gym

"""We use the pre-built environment from OpenAI Gym.
To support the use of the experiment class we create a small custom class
that wraps the Gym environment. 

Documentation: https://www.gymlibrary.dev/environments/toy_text/blackjack/
"""

class Blackjack:
    def __init__(self, args):
        natural = args.get('natural', False)
        sab = args.get('sab', False)
        self.env = gym.make('Blackjack-v1', natural=natural, sab=sab)
        self.env_dimensions = tuple(dim.n for dim in self.env.observation_space.spaces)
        self.action_space = self.env.action_space
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def step(self, a):
        """
        Actions:
         - 0: stick
         - 1: hit
        """
        obs, r, terminated, _, _ = self.env.step(a)
        return obs, r, terminated
