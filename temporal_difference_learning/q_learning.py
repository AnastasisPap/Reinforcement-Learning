from __future__ import annotations
import gymnasium as gym
import numpy as np

from libs.utils.agent import BaseAgent

class QLearning(BaseAgent):
    def __init__(self, env: gym.Env, args: dict) -> None:
        super().__init__(env, args)

    def step(self, s: tuple | int) -> tuple[tuple | int, bool, int]:
        a = self.eps_greedy(s)
        next_s, r, is_terminal = self.env.step(a)

        self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

        return next_s, is_terminal, r