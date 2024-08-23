import numpy as np

from libs.utils.agent import BaseAgent

class ValueIteration(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.theta = args.get('theta', 1e-6)

        # The dynamics must be known for MDP algorithms
        assert hasattr(env, 'get_dynamics'), 'MDPs require environment dynamics to be known.'
        assert hasattr(env, 'observation_space'), 'MDPs require environment observation space to be known.'
        self.observation_space = env.observation_space
        self.dynamics = env.get_dynamics()
        self.policy = {s: 0 for s in self.observation_space}
        self.sweeps = []

    def iteration(self):
        delta = float('inf')

        while delta >= self.theta:
            delta = 0

            for s in self.observation_space[1:-1]:
                prev_value = self.V[s]

                max_v = float('-inf')
                for a in self.env.get_actions(s):
                    v = sum(
                        [
                            prob * (r + self.gamma * self.V[s_prime])
                            for (s_prime, r), prob in self.dynamics[s, a].items()
                        ])
                    
                    if v > max_v:
                        max_v = v
                self.V[s] = max_v

                delta = max(delta, abs(prev_value - self.V[s]))
            self.sweeps.append(self.V.copy())
        
        for s in self.observation_space[1:-1]:
            max_v = float('-inf')
            values = np.zeros(len(self.env.get_actions(s)))

            for i, a in enumerate(self.env.get_actions(s)[1:]):
                values[i] = sum(
                    [
                        prob * (r + self.gamma * self.V[s_prime])
                        for (s_prime, r), prob in self.dynamics[s, a].items()
                    ])
                
            self.policy[s] = np.argmax(values)
        
        return True