from __future__ import annotations
import pickle
import numpy as np

from libs.utils.experiment import experiment
from libs.utils.graphing import graph_policy_trajectory, plot_3d, plot_policy

from libs.envs.blackjack import Blackjack
from libs.envs.racetrack import RacetrackEnv

from monte_carlo.first_visit_prediction import FirstVisitMC
from monte_carlo.exploring_starts import ExploringStarts
from monte_carlo.off_policy_mc_control import OffPolicyMCControl

def create_grids(
        data: dict | np.ndarray,
        usable_ace: int=0
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(np.arange(12, 22), np.arange(1, 11))
    Z = np.zeros(X.shape)
    for i in range(12, 22):
        for j in range(1, 11):
            Z[i-12, j-1] = data[(i, j, usable_ace)]
    return X, Y, Z

def create_policy_grid(
        policy: dict | np.ndarray,
        usable_ace: int=0
        ) -> np.ndarray:
    X, Y = np.meshgrid(np.arange(12, 22), np.arange(1, 11))
    Z = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([X, Y])
    )

    return Z

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
        'No usable ace', './monte_carlo/results/5_1_no_usable.png', {})

    plot_3d(
        X_1, Y_1, Z_1,
        'Player sum', 'Dealer showing', 'State-value',
        'No usable ace', './monte_carlo/results/5_1_usable.png', {})

def experiment_5_2():
    print('Starting experiment for example 5.2')

    exp_args = {'episodes': 500000}
    env_args = {'sab': True, 'states_dim': (32, 11, 2)}
    
    data = experiment(Blackjack, ExploringStarts, env_args, {}, exp_args)

    values = data['estimated_values_per_episode']
    X_0, Y_0, Z_0 = create_grids(values[-1])
    X_1, Y_1, Z_1 = create_grids(values[-1], 1)
    policy_grid_0 = create_policy_grid(data['policy_per_rep'][0])
    policy_grid_1 = create_policy_grid(data['policy_per_rep'][0], 1)

    plot_3d(
        X_0, Y_0, Z_0,
        'Player sum', 'Dealer showing', 'State-value',
        'No usable ace', './monte_carlo/results/5_2_no_usable.png', {})

    plot_3d(
        X_1, Y_1, Z_1,
        'Player sum', 'Dealer showing', 'State-value',
        'No usable ace', './monte_carlo/results/5_2_usable.png', {})

    args = {'x_ticks': list(range(12, 22)), 'y_ticks': ['A'] + list(range(2, 11)), 'cbar_ticks': [0, 1], 'cbar_labels': ['Stick', 'Hit']}
    plot_policy(
        policy_grid_0,
        'Player sum', 'Dealer showing', 'No usable ace',
        './monte_carlo/results/5_2_policy_no_usable.png', args
    )

    plot_policy(
        policy_grid_1,
        'Player sum', 'Dealer showing', 'Usable ace',
        './monte_carlo/results/5_2_policy_usable.png', args
    )

def experiment_5_12():
    print('Starting experiment for exercise 5.12')

    exp_args = {'episodes': 500_000}
    agent_args = {'gamma': 0.9, 'init_value': 'normal'}

    data = experiment(RacetrackEnv, OffPolicyMCControl, {}, agent_args, exp_args)
    with open('./monte_carlo/results/racetrack_policy_500k_2.pkl', 'wb') as f:
        pickle.dump(data, f)


    """If there is a prestored pickle data file, then uncomment the following and 
    comment the previous block of code.
    """
    #data = pickle.load(open('./monte_carlo/results/racetrack_policy_500k.pkl', 'rb'))
    policy = data['policy_per_rep'][0]

    graph_policy_trajectory(RacetrackEnv, policy, {}, './monte_carlo/results/5_12_racetrack_trajectory.png')

if __name__ == '__main__':
    experiment_5_12()