import numpy as np
from libs.envs.dyna_maze import DynaMaze
from libs.utils.graphing import plot_results
from planning_and_learning.dynaQ import DynaQAgent
from tqdm import tqdm

class DynaQPlusAgent(DynaQAgent):
    def __init__(self, env, agent_args):
        super().__init__(env, agent_args)
        self.kappa = agent_args.get('kappa', 0.001)
        self.transition_history = np.zeros((env.height, env.width, env.action_space.n))

    def planning(self):
        for _ in range(self.n):
            s, a = self.model.sample()
            next_s, reward = self.model.get(s, a)
            reward += self.kappa * np.sqrt(self.transition_history[s][a])
            self.Q[s][a] += self.alpha * (reward + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

    def step(self, s):
        a = self.eps_greedy(s)
        next_s, r, is_terminal = self.env.step(a)
        self.transition_history += 1
        self.transition_history[s][a] = 0

        self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

        self.model.update(s, a, r, next_s)
        self.planning()

        return next_s, is_terminal, r