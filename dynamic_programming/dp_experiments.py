import pickle
import numpy as np
import matplotlib.pyplot as plt

from libs.utils.experiment import dp_experiment, load_experiment
from libs.utils.graphing import plot_policy, plot_3d, plot_results

from libs.envs.car_rental import CarRentalEnv
from libs.envs.coin_flip_gamble import CoinFlipGambleEnv

from dynamic_programming.mdp_policy_iteration import MDPPolicyIter
from dynamic_programming.value_iteration import ValueIteration

def experiment_4_2():
    agent_args = {'gamma': 0.9, 'load_dynamics': True, 'dynamics_path': './dynamic_programming/results/dynamics.pkl'}
    exp_args = {'store_data': True, 'data_path': './dynamic_programming/results/experiment_4_2_data.pkl'}

    data = dp_experiment(CarRentalEnv, MDPPolicyIter, {}, agent_args, exp_args)
    #data = load_experiment('./dynamic_programming/results/experiment_4_2_data.pkl')

    policies = data['policies']
    for i in range(len(policies)):
        policy_grid = np.zeros((21, 21))
        for s, v in policies[i].items():
            policy_grid[s[0], s[1]] = v

        args = {'cmap': 'magma', 'origin': 'lower'}
        plot_policy(
            policy_grid,
            '# Cars at second location', '# Cars at first location',
            f'Policy at iteration {i+1}', f'./dynamic_programming/results/experiment_4_2_policy_{i+1}.png',
            args
        )

    X, Y = np.meshgrid(np.arange(21), np.arange(21))
    plot_3d(
        X, Y, data['value'],
        '# of cars at 2nd location', '# of cars at 1st location', 'Value',
        'Value function', './dynamic_programming/results/experiment_4_2_value.png', {}
    )

def experiment_4_3():
    data = dp_experiment(CoinFlipGambleEnv, ValueIteration, {}, {'dtype': np.float128, 'theta': 1e-16}, {})

    sweeps = data['sweeps'][-1]
    y = [sweep[1:-1] for sweep in sweeps]
    y = [y[0], y[1], y[2], y[3], y[-1]]
    labels = ['Sweep 1', 'Sweep 2', 'Sweep 3', 'Sweep 4', f'Sweep {len(sweeps)+1}']
    x = np.arange(1, len(y[0])+1)

    plot_results(
        x, y, 'Capital', 'Value', labels,
        './dynamic_programming/results/experiment_4_3.png'
    )

    policy = data['policies'][-1]
    y = [policy[s] for s in x]

    plot_results(
        x, [y], 'Capital', 'Final policy', ['Policy'],
        './dynamic_programming/results/experiment_4_3_policy.png'
    )

if __name__ == '__main__':
    experiment_4_3()