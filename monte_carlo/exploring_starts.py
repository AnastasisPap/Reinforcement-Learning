import itertools
import numpy as np

#from monte_carlo.first_visit_prediction import FirstVisitMC
from libs.utils.agent import BaseAgent

class ExploringStarts(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.returns = {}
        self.init_policy()

    def init_policy(self):
        self.policy = {}
        dim_ranges = [range(dim.n) for dim in self.env.observation_space]
        states = list(itertools.product(*dim_ranges))

        for s in states:
            self.policy[s] = 1 if s[0] <= 20 else 0

    def generate_episode(self, s):
        states = [s]
        rewards = []
        actions = []
        is_term = False

        while not is_term:
            a = self.policy[s] if len(actions) > 0 else np.random.randint(self.env.action_space.n)
            s, r, is_term = self.env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
        
        return states[:-1], actions, rewards


    def step(self, s):
        states, actions, rewards = self.generate_episode(s)

        G = 0.0
        T = len(rewards)

        for t in range(T-1, -1, -1):
            G = self.gamma * G + rewards[t]
            s, a = states[t], actions[t]

            if (s, a) not in zip(states[:t], actions[:t]):
                if (s, a) not in self.returns:
                    self.returns[(s, a)] = []
                self.returns[(s, a)].append(G)
                self.Q[s][a] = np.mean(self.returns[(s, a)])
                self.policy[s] = np.argmax(self.Q[s])
                self.V[s] = np.max(self.Q[s])
        
        return None, True, 0