from __future__ import annotations

import pickle
import gymnasium as gym
import numpy as np

from scipy.stats import poisson
from gymnasium import spaces
from tqdm import tqdm

class CarRentalEnv(gym.Env):
    def __init__(self, args: dict) -> None:
        lambda_first_req, lambda_first_ret = args.get('lambda_first', (3, 3))
        lambda_second_req, lambda_second_ret = args.get('lambda_second', (4, 2))
        self.max_cars = args.get('max_cars', 20)
        self.max_cars_moved = args.get('max_cars_moved', 5)

        self.credit = args.get('credit', 10)
        self.cost = args.get('cost', 2)

        self.env_dimensions = (self.max_cars + 1, self.max_cars + 1)
        self.action_space = spaces.Discrete(2 * self.max_cars_moved + 1)

        # For some action a:
        # if a is negative, then |a| cars are moved from the first to the second
        # if a is positive, then a cars are from from the second to the first
        self.actions = list(range(-self.max_cars_moved, self.max_cars_moved + 1))
        
        self.observation_space = [(i, j) for i in range(self.max_cars + 1) for j in range(self.max_cars + 1)]

        self.min_prob = args.get('min_prob', 1e-6)
        self.probs = self.get_probs(
            [lambda_first_req, lambda_second_req, lambda_first_ret, lambda_second_ret], self.min_prob)
    
    def get_probs(self, lambdas, min_prob):
        req_ret_space = np.array(
            [[i, j, k, l]
             for i in range(self.max_cars + 1)
             for j in range(self.max_cars + 1)
             for k in range(self.max_cars + 1)
             for l in range(self.max_cars + 1)])

        probs = poisson.pmf(req_ret_space, lambdas).prod(axis=1)
        probs = {tuple(req_ret_space[i]): probs[i] for i in range(len(req_ret_space))}
        return {transition: prob for transition, prob in probs.items() if prob > min_prob}
    
    def check_state_action(self, s, a):
        if a > 0: return s[0] >= a
        elif a < 0: return s[1] >= -a

        return True
    
    def get_dynamics(
            self,
            file_path:str = './libs/envs/dynamics.pkl',
            load:bool = False
        ) -> dict[dict[float]]:
        """For all the possible valid state-action pairs, calculate the dynamics of the environment.
        So given a state-action pair (s,a), the dynamics are the probabilities of transitioning to
        a new state s' and getting reward r. So for a given initial (s, a) and the transition to s' and r
        the result is the probability of that transition happening.

        Args:
            file_path (str, optional): Path to save the dynamics. Defaults to './libs/envs/dynamics.pkl'.
            load (bool, optional): Whether to load the dynamics from file. Defaults to False.
        Returns:
            dict[dict[float]]: The dynamics of the environment.
        """
        if load:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                print('Loaded dynamics.')
                return data

        dynamics = {}

        for s in tqdm(self.observation_space):
            for a in self.actions:
                if not self.check_state_action(s, a):
                    continue
                
                curr_dynamics = {}
                s_prime = (s[0] - a, s[1] + a)
                cost = self.cost * abs(a)

                for (first_req, second_req, first_ret, second_ret), prob in self.probs.items():
                    requests = np.min([s_prime, [first_req, second_req]], axis=0)
                    curr_cars = np.min([[self.max_cars, self.max_cars], s_prime-requests], axis=0)
                    returns = np.min([[first_ret, second_ret], self.max_cars-curr_cars], axis=0)
                    curr_cars += returns

                    r = np.sum(requests) * self.credit - cost
                    new_s = tuple(curr_cars)
                    if (new_s, r) not in curr_dynamics:
                        curr_dynamics[(new_s, r)] = 0
                    curr_dynamics[(new_s, r)] += prob
                    
                dynamics[(s, a)] = curr_dynamics
        
        with open(file_path, 'wb') as f:
            pickle.dump(dynamics, f)

        return dynamics