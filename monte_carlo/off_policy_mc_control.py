from __future__ import annotations
import gymnasium as gym
import numpy as np

from libs.utils.agent import BaseAgent

class OffPolicyMCControl(BaseAgent):
    def __init__(self, env: gym.Env, args: dict) -> None:
        super().__init__(env, args)
        self.Q -= 500
        # Behavior policy must return a tuple (action, probability of choosing that action)
        self.b = args.get('b_policy', self.behavior_policy)
        self.policy = np.argmax(self.Q, axis=-1)
        self.C = np.zeros_like(self.Q)

    def behavior_policy(self, s: tuple | int) -> tuple[int, float]:
        if np.random.uniform() > self.epsilon:
            return self.policy[s], 1-self.epsilon+self.epsilon/self.env.action_space.n
        
        return np.random.randint(self.env.action_space.n), self.epsilon/self.env.action_space.n
    
    def generate_episode(self, s: tuple | int) -> list[tuple]:
        trajectory = []

        is_term = False
        a, a_prob = self.b(s)

        while not is_term:
            next_s, r, is_term = self.env.step(a)
            trajectory.append((s, a, r, a_prob))
            s = next_s
            if not is_term: a, a_prob = self.b(s)

        return trajectory
    
    def step(self, s: tuple | int) -> tuple[None, bool, int]:
        trajectory = self.generate_episode(s)

        G = 0.0
        W = 1.0

        for t in range(len(trajectory)-1, -1, -1):
            s, a, r, a_prob = trajectory[t]

            G = self.gamma * G + r
            self.C[s][a] += W
            self.Q[s][a] += (W / self.C[s][a]) * (G - self.Q[s][a])

            self.policy[s] = np.argmax(self.Q[s])
            if a != self.policy[s]:
                break

            W = W / a_prob

        return None, True, 0