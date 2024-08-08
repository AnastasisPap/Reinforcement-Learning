import numpy as np


from libs.utils.experiment import experiment
from libs.utils.graphing import plot_results

from libs.envs.random_walk import RandomWalk
from libs.envs.windy_grid_world import WindyGridWorld
from libs.envs.cliff_walk import CliffWalk

from temporal_difference_learning.td_zero import TDZero
from temporal_difference_learning.sarsa import Sarsa
from temporal_difference_learning.q_learning import QLearning

def experiment_6_2():
    print('Starting experiment for example 6.2')

    true_values = np.arange(7) / 6.0
    true_values[0] = 0
    true_values[-1] = 0
    exp_args = {'episodes': 100, 'repetitions': 100, 'true_values': true_values}
    env_args = {'n': 5, 'states_dim': 7, 'seed': 0}
    agent_args = {'init_value': 0.5}

    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
    data = []
    for alpha in alphas:
        agent_args['alpha'] = alpha
        data.append(experiment(RandomWalk, TDZero, env_args, agent_args, exp_args))

    rms_errors = []
    for i in range(len(alphas)):
        rms_errors.append(np.sqrt(np.sum(np.power(data[i]['estimated_values_per_episode'] - true_values, 2), axis=1) / 5))
    plot_results(
        np.arange(exp_args['episodes']),
        rms_errors,
        x_label='Walks/Episodes',
        y_label='RMS Error',
        labels=[f'a={alpha}' for alpha in alphas],
        file_path='./temporal_difference_learning/results/td_zero_6_2_rms.png'
    )

    estimated_values = data[5]['estimated_values_per_episode'][:, 1:-1]
    results = estimated_values[[0, 9, 99], :]
    results = np.vstack([np.full(5, 0.5), results, true_values[1:-1]])
    plot_results(
        np.arange(1, 6),
        results,
        x_label='State',
        y_label='Estimated Value',
        labels=['0 Episodes', '1 Episode', '10 Episodes', '100 Episodes', 'True Values'],
        file_path='./temporal_difference_learning/results/td_zero_6_2.png'
    )

def experiment_6_5():
    print('Starting experiment for example 6.5')

    agent_args = {'alpha': 0.5}
    env_args = {'dimensions': (7, 10), 'start_state': (3, 0), 'goal_state': (3, 7)}
    exp_args = {'max_steps': 8000}

    data = experiment(WindyGridWorld, Sarsa, env_args, agent_args, exp_args)

    plot_results(
        np.arange(exp_args['max_steps']),
        [data['episodes_per_time_step']],
        x_label='Time Steps',
        y_label='Episodes',
        labels=['Sarsa'],
        file_path='./temporal_difference_learning/results/sarsa_6_5.png'
    )

def experiment_6_6():
    print('Starting experiment for example 6.6')

    agent_args = {'alpha': 0.5}
    env_args = {'dimensions': (4, 12), 'start_state': (3, 0), 'goal_state': (3, 11)}
    exp_args = {'episodes': 500, 'repetitions': 100}
    
    data = []
    data.append(experiment(CliffWalk, Sarsa, env_args, agent_args, exp_args)['cum_reward_per_episode'])
    data.append(experiment(CliffWalk, QLearning, env_args, agent_args, exp_args)['cum_reward_per_episode'])

    plot_results(
        np.arange(exp_args['episodes']),
        data,
        x_label='Episodes',
        y_label='Sum of rewards during episode',
        labels=['Sarsa', 'Q-Learning'],
        file_path='./temporal_difference_learning/results/qlearning_6_6.png'
        [-100, -25]
    )

if __name__ == '__main__':
    experiment_6_6()
