import numpy as np
import gymnasium as gym

from libs.utils.experiment import experiment
from libs.utils.graphing import plot_results, plot_3d

from libs.envs.blackjack import Blackjack

from monte_carlo.first_visit_prediction import FirstVisitMC

def create_grids(data, usable_ace=0):
    X, Y = np.meshgrid(np.arange(12, 22), np.arange(1, 11))
    Z = np.zeros(X.shape)
    for i in range(12, 22):
        for j in range(1, 11):
            Z[i-12, j-1] = data[(i, j, usable_ace)]
    return X, Y, Z

def experiment_5_1():
    print('Starting experiment for example 5.1')

    exp_args = {'episodes': 500000}
    env_args = {'sab': True, 'states_dim': (32, 11, 2)}
    
    policy = lambda s: 0 if s[0] >= 20 else 1 
    data = experiment(Blackjack, FirstVisitMC, env_args, {'policy': policy}, exp_args)

    data = data['estimated_values_per_episode']
    X_0, Y_0, Z_0 = create_grids(data[-1])
    X_1, Y_1, Z_1 = create_grids(data[-1], 1)

    plot_3d(
        X_0, Y_0, Z_0,
        'Player sum', 'Dealer showing', 'State-value',
        'No usable ace', './monte_carlo/results/5_1_no_usable.png')

    plot_3d(
        X_1, Y_1, Z_1,
        'Player sum', 'Dealer showing', 'State-value',
        'No usable ace', './monte_carlo/results/5_1_usable.png')

if __name__ == '__main__':
    experiment_5_1()