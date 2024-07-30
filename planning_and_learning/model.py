import numpy as np

class Model:
    """
    Args:
        Env (Gymnasium Env)
        Seed (int)
    """
    def __init__(self, env, seed):
        self.model = {}
        # To avoid random anomalities, we use a random seed for the same
        # episodes
        self.rnd_gen = np.random.RandomState(seed)
        self.env = env
    
    """Updates the model in which when the agent interacted with the environment
    and chose action "a" when at the state "s", the agent transitioned to the state
    "next_s" and observed a reward "r".

    Args:
        s (tuple of ints): initial state
        a (int): action taken in state s
        next_s (tuple of ints): state after taking action a
        r (float): observed reward
    """
    def update(self, s, a, r, next_s):
        if s not in self.model: self.model[s] = {}
        self.model[s][a] = (next_s, r)
    
    """Randomly selects an observed (non-terminal) state s, and a randomly observed action a
    taken when in state s.

    Returns:
        s (tuple of ints): random observed (non-terminal) state
        a (int): random observed action when in state s
    """
    def sample(self):
        states = list(self.model.keys())
        s_random = states[self.rnd_gen.randint(len(states))]
        while self.env.is_terminal(s_random):
            s_random = states[self.rnd_gen.randint(len(states))]

        actions = list(self.model[s_random].keys())
        a_random = actions[self.rnd_gen.randint(len(actions))]

        return s_random, a_random
    
    def get(self, s, a):
        return self.model[s][a]