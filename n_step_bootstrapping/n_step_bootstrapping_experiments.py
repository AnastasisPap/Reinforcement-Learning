import numpy as np

from libs.envs.random_walk import RandomWalk
from libs.utils.experiment import experiment
from libs.utils.graphing import plot_results
from n_step_bootstrapping.n_step_td import NStepTD

def experiment_7_1():
    print('Starting experiment for example 7.1')

    alphas = np.arange(0, 1.1, 0.1)
    steps = 2 ** np.arange(10)
    env_args = {'n': 19, 'left_r': -1}

    true_values = np.arange(-20, 22, 2) / 20.0
    true_values[0] = 0
    true_values[-1] = 0
    exp_args = {'episodes': 10, 'repetitions': 100, 'true_values': true_values}

    td_error = np.zeros((len(steps), len(alphas)))

    for i, step in enumerate(steps):
        for j, alpha in enumerate(alphas):
            data = experiment(RandomWalk, NStepTD, env_args, {'alpha': alpha, 'n': step}, exp_args)
            td_error[i, j] = data['rms_error']
    
    td_error /= exp_args['episodes'] * exp_args['repetitions']

    plot_results(
        alphas,
        td_error,
        x_label='alpha',
        y_label='Average RMS error',
        labels=[f'n={step}' for step in steps],
        file_path='./experiments/n_step_bootstrapping/results/n_step_td_7_1.png'
    )

if __name__ == '__main__':
    experiment_7_1()