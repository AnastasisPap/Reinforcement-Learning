import numpy as np
import heapq

from libs.utils.agent import BaseAgent
from planning_and_learning.model import Model

class PrioritizedDynaQ(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rnd_gen = np.random.RandomState(args.get('seed', 0))
        self.theta = args.get('theta', 0.001)
        self.n = args.get('n', 10)
        self.model = Model(env, self.rnd_gen)
        self.pq = []

    def prioritized_planning(self):
        i = 0
        while i < self.n and len(self.pq):
            _, s, a = heapq.heappop(self.pq)
            next_s, r = self.model.get(s, a)
            self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

            for s_hat, a_hat in self.model.get_predecessors(s):
                _, r = self.model.get(s_hat, a_hat)
                P = abs(r + self.gamma * np.max(self.Q[s]) - self.Q[s_hat][a_hat])
                if P > self.theta:
                    heapq.heappush(self.pq, (-P, s_hat, a_hat))
            
            i += 1

    def step(self, s):
        a = self.eps_greedy(s)

        next_s, r, is_terminal = self.env.step(a)
        self.model.update(s, a, r, next_s)

        P = abs(r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])
        if P > self.theta:
            heapq.heappush(self.pq, (-P, s, a))
        
        self.prioritized_planning()

        return next_s, is_terminal, r