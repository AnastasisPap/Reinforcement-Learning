import numpy as np

from libs.utils.graphing import plot_results
from libs.utils.experiment import bandit_experiment

from libs.envs.armed_testbed import ArmedTestbed

from multi_armed_bandits.simple_bandit import SimpleBandit

def experiment_2_2():
    data = [
        bandit_experiment(ArmedTestbed, SimpleBandit, {}, {'epsilon': 0.0}, {}),
        bandit_experiment(ArmedTestbed, SimpleBandit, {}, {'epsilon': 0.1}, {}),
        bandit_experiment(ArmedTestbed, SimpleBandit, {}, {'epsilon': 0.01}, {})
    ]
    labels = ['Greedy', 'Epsilon-Greedy 0.1', 'Epsilon-Greedy 0.01']

    x = np.arange(1000)
    avg_rewards = [i['avg_rewards'] for i in data]
    plot_results(
        x, avg_rewards,
        'Steps', 'Avg reward', labels,
        './multi_armed_bandits/results/simple_bandit_avg_rewards.png'
    )

    chosen_opt = [i['chosen_opt'] for i in data]
    plot_results(
        x, chosen_opt,
        'Steps', '% Optimal action', labels,
        './multi_armed_bandits/results/simple_bandit_opt_perc.png'
    )

if __name__ == '__main__':
    experiment_2_2()