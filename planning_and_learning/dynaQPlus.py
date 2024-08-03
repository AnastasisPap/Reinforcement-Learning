import numpy as np
from libs.envs.dyna_maze import DynaMaze
from libs.graphs.graphing import plot_results
from planning_and_learning.dynaQ import DynaQAgent
from tqdm import tqdm

class DynaQPlusAgent(DynaQAgent):
    def __init__(self, env, agent_args):
        super().__init__(env, agent_args)
        self.kappa = agent_args.get('kappa', 0.001)
        self.transition_history = np.zeros((env.height, env.width, env.action_space.n))

    def planning(self):
        for _ in range(self.n):
            s, a = self.model.sample()
            next_s, reward = self.model.get(s, a)
            reward += self.kappa * np.sqrt(self.transition_history[s][a])
            self.Q[s][a] += self.alpha * (reward + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

    def step(self, s):
        a = self.eps_greedy(s)
        next_s, r, is_terminal = self.env.step(a)
        self.transition_history += 1
        self.transition_history[s][a] = 0

        self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

        self.model.update(s, a, r, next_s)
        self.planning()

        return next_s, is_terminal, r

def experiment(env_class, agent_class, env_info, agent_info, exp_parameters):
    num_runs = exp_parameters['num_runs']
    num_max_steps = exp_parameters['num_max_steps']
    cum_reward_all = np.zeros((num_runs, num_max_steps))

    for run in tqdm(range(num_runs)):
        env = env_class(env_info)
        agent_info['seed'] = run
        agent = agent_class(env, agent_info)

        num_steps = 0
        cum_reward = 0

        while num_steps < num_max_steps-1 :
            s = env.reset()
            is_terminal = False

            while not is_terminal and num_steps < num_max_steps-1 :
                s, is_terminal, r = agent.step(s)

                num_steps += 1
                cum_reward += r
                cum_reward_all[run][num_steps] = cum_reward

    return np.mean(cum_reward_all, axis=0)

if __name__ == '__main__':
    experiment_args = {'num_runs' : 10, 'num_max_steps' : 3000}
    environment_args = {
        'iter_change': 1000, 'start_state': (5, 3),
        'obstacles': [(3, i) for i in range(8)],
        'obstacles_after_change': [(3, i) for i in range(1, 9)]}
    agent_args = {'epsilon': 0.1, 'alpha' : 0.5, 'gamma': 0.95, 'n' : 50}

    agents_cum_rewards = []
    agents_cum_rewards.append(experiment(DynaMaze, DynaQAgent, environment_args, agent_args, experiment_args))
    agents_cum_rewards.append(experiment(DynaMaze, DynaQPlusAgent, environment_args, agent_args, experiment_args))

    plot_results(
        np.arange(1, experiment_args['num_max_steps'] + 1),
        agents_cum_rewards,
        x_label='Time steps',
        y_label='Cumulative rewards',
        labels=['Dyna-Q', 'Dyna-Q+'],
        file_path='./libs/graphs/results/dynaQplus_maze_8_2.png'
    )