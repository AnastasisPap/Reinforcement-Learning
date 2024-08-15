from __future__ import annotations
import gymnasium as gym
import numpy as np

from libs.utils.agent import BaseAgent

class FirstVisitMC(BaseAgent):
    def __init__(self, env: gym.Env, args: dict) -> None:
        super().__init__(env, args)
        self.policy = args.get('policy', self.eps_greedy)
        self.returns = {}

    def generate_episode(self, s: tuple | int) -> tuple[list, list]:
        states = [s]
        rewards = []
        is_terminal = False

        while not is_terminal:
            a = self.policy(s)
            s, r, is_terminal = self.env.step(a)
            states.append(s)
            rewards.append(r)
        
        return states[:-1], rewards
    
    def step(self, s: tuple | int) -> tuple[None, bool, int]:
        gen_states, gen_rs = self.generate_episode(s)

        G = 0.0
        T = len(gen_rs)

        for t in range(T-1, -1, -1):
            G = self.gamma * G + gen_rs[t]
            s = gen_states[t]
            if s not in gen_states[:t]:
                if s not in self.returns:
                    self.returns[s] = []
                self.returns[s].append(G)
                self.V[s] = np.mean(self.returns[s])
        
        return None, True, 0