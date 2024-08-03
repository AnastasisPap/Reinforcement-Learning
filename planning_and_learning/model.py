import numpy as np

class Model:
    def __init__(self, env, rnd_gen):
        """
        Args:
            Env (Gymnasium Env)
            Seed (int)
        """
        self.model = {}
        # To avoid random anomalities, we use a random seed for the same
        # episodes
        self.rnd_gen = rnd_gen
        self.env = env
    
    def update(self, s, a, r, next_s):
        """Updates the model in which when the agent interacted with the environment
        and chose action "a" when at the state "s", the agent transitioned to the state
        "next_s" and observed a reward "r".

        Args:
            s (tuple of ints): initial state
            a (int): action taken in state s
            next_s (tuple of ints): state after taking action a
            r (float): observed reward
        """
        if s not in self.model:
            self.model[s] = {}
            for new_a in range(self.env.action_space.n):
                self.model[s][new_a] = (s, 0)
        self.model[s][a] = (next_s, r)
    
    def sample(self):
        """Randomly selects an observed (non-terminal) state s, and a randomly observed action a
        taken when in state s.

        Returns:
            s (tuple of ints): random observed (non-terminal) state
            a (int): random observed action when in state s
        """
        s_keys = list(self.model.keys())
        s_random = s_keys[self.rnd_gen.randint(len(s_keys))]

        a_keys = list(self.model[s_random].keys())
        a_random = a_keys[self.rnd_gen.randint(len(a_keys))]

        return s_random, a_random
    
    def get(self, s, a):
        return self.model[s][a]