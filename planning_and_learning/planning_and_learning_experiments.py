import numpy as np

from libs.envs.dyna_maze import DynaMaze
from libs.utils.graphing import plot_results
from planning_and_learning.dynaQ import DynaQAgent
from planning_and_learning.dynaQPlus import DynaQPlusAgent
from libs.utils.experiment import experiment

def experiment_8_1():
    print('Starting experiment for example 8.1')
    n_values = [0, 5, 50]

    experiment_args = {'episodes': 50, 'repetitions': 30}
    num_of_steps = np.zeros((len(n_values), experiment_args['episodes']-1))
    agent_args = {'n': 0, 'gamma': 0.95, 'alpha': 0.1, 'epsilon': 0.1}
    env_args = {
        'start_state': (2, 0), 'goal_state': (0, 8),
        'obstacles': [(i, 2) for i in range(1, 4)] + [(i, 7) for i in range(0, 3)] + [(4,5)]
        }

    for i, n in enumerate(n_values):
        agent_args['n'] = n
        data = experiment(DynaMaze, DynaQAgent, env_args, agent_args, experiment_args)
        num_of_steps[i] = np.mean(data['avg_steps_per_episode'], axis=0)[1:]
    
    plot_results(
        np.arange(2, experiment_args['episodes'] + 1),
        num_of_steps,
        'Episodes',
        'Steps per Episode',
        [f'n={n}' for n in n_values],
        './experiments/planning_and_learning/results/dynaQ_8_1.png'
    )

def experiment_8_2():
    print('Starting experiment for example 8.2')
    experiment_args = {'repetitions': 10, 'max_steps' : 3000}
    environment_args = {
        'iter_change': 1000, 'start_state': (5, 3),
        'obstacles': [(3, i) for i in range(8)],
        'obstacles_after_change': [(3, i) for i in range(1, 9)]}
    agent_args = {'epsilon': 0.1, 'alpha' : 0.5, 'gamma': 0.95, 'n' : 50}

    agents_cum_rewards = []
    data_1 = experiment(DynaMaze, DynaQAgent, environment_args, agent_args, experiment_args)['cum_reward']
    data_2 = experiment(DynaMaze, DynaQPlusAgent, environment_args, agent_args, experiment_args)['cum_reward']
    agents_cum_rewards.append(np.mean(data_1, axis=0))
    agents_cum_rewards.append(np.mean(data_2, axis=0))

    plot_results(
        np.arange(1, experiment_args['max_steps'] + 1),
        agents_cum_rewards,
        x_label='Time steps',
        y_label='Cumulative rewards',
        labels=['Dyna-Q', 'Dyna-Q+'],
        file_path='./experiments/planning_and_learning/results/dynaQ_8_2.png'
    )


def experiment_8_3():
    print('Starting experiment for example 8.3')
    experiment_args = {'repetitions': 10, 'max_steps' : 6000}
    environment_args = {
        'iter_change': 3000, 'start_state': (5, 3),
        'obstacles': [(3, i) for i in range(1, 9)],
        'obstacles_after_change': [(3, i) for i in range(1, 8)]}
    agent_args = {'epsilon': 0.1, 'alpha' : 0.5, 'gamma': 0.95, 'n' : 50}

    agents_cum_rewards = []
    data_1 = experiment(DynaMaze, DynaQAgent, environment_args, agent_args, experiment_args)['cum_reward']
    data_2 = experiment(DynaMaze, DynaQPlusAgent, environment_args, agent_args, experiment_args)['cum_reward']
    agents_cum_rewards.append(np.mean(data_1, axis=0))
    agents_cum_rewards.append(np.mean(data_2, axis=0))

    plot_results(
        np.arange(1, experiment_args['max_steps'] + 1),
        agents_cum_rewards,
        x_label='Time steps',
        y_label='Cumulative rewards',
        labels=['Dyna-Q', 'Dyna-Q+'],
        file_path='./experiments/planning_and_learning/results/dynaQ_8_3.png'
    )


if __name__ == '__main__':
    experiment_8_1()
    experiment_8_2()
    experiment_8_3()
