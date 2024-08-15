from __future__ import annotations
import gymnasium as gym

from libs.utils.agent import BaseAgent

class TDZero(BaseAgent):
    def __init__(self, env: gym.Env, args: dict) -> None:
        super().__init__(env, args)
        self.V[0] = 0.0
        self.V[-1] = 0.0
    
    def reset(self, s: tuple | int) -> None:
        self.s = s
    
    def step(self, s: tuple | int) -> tuple[tuple | int, bool]:
        a = self.eps_greedy(s)
        next_s, r, is_term = self.env.step(a)
        self.V[s] += self.alpha * (r + self.gamma * self.V[next_s] - self.V[s])

        return next_s, is_term, r