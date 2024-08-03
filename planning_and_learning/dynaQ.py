import sys, os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

import numpy as np
from libs.envs.dyna_maze import DynaMaze
from libs.graphs.graphing import plot_results
from planning_and_learning.model import Model
from tqdm import tqdm

class DynaQAgent:
    def __init__(self, env, args):
        """
        Args:
            Env: object of DynaMaze which is the agent environment
            args: dictionary mapping name (str) -> value (int or float)
        """
        self.epsilon = args.get('epsilon', 0.1)
        self.gamma = args.get('gamma', 0.95)
        self.alpha = args.get('alpha', 0.1)
        self.n = args.get('n', 10)
        self.env = env

        self.rnd_gen = np.random.RandomState(args.get('seed', 0))
        self.Q = np.zeros((self.env.height, self.env.width, self.env.action_space.n))
        self.model = Model(env, self.rnd_gen)

    def eps_greedy(self, s):
        """It's the epsilon greedy policy. With probability epsilon selects any action
        randomly with the same probability, and with prob 1-epsilon, greedily selects
        the action and breaks ties arbitrarily.

        Args:
            The state (tuple of ints) for which the policy will be used.
        Returns:
            The action (int) which is an int in [0, 3]) which the policy chose.
        """
        prob = self.rnd_gen.uniform()
        if prob < self.epsilon:
            return self.rnd_gen.randint(self.env.action_space.n)
        else:
            return self.rnd_gen.choice(np.where(np.max(self.Q[s]) == self.Q[s])[0])
    
    def planning(self):
        """For Dyna-Q agents with n>0, the following is the planning step.
        """
        for i in range(self.n):
            s, a = self.model.sample()
            next_s, r = self.model.get(s, a)
            self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

    def step(self, s):
        """Completes one step during an episode (transitions from time t to t+1). The agent uses
        the epsilon greedy policy to choose an action for the current state, it takes the action
        and observes the new state and reward. Then updates the action-value function Q and the model.
        Lastly, if there are planning steps, it also does planning.

        Args:
            The current state (tuple of ints) the agent is in
        Returns:
            State (tuple of ints): the next state which the agent transitioned to
            Is terminal (bool): if the new state is terminal, this is true
        """
        a = self.eps_greedy(s)

        next_s, r, is_terminal = self.env.step(a)

        self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])
        self.model.update(s, a, r, next_s)

        self.planning()
        
        return next_s, is_terminal, r
    
def experiment(env_args, agent_args, experiment_args):
    """Completes the experiment. To properly calculate the number of steps required to complete
    the episode, multiple iterations are used and then averaged. For each iteration, multiple
    episodes are used. At the start of each new run, the agent is reset, while the env is
    reset every new episode.

    Args:
        Agent (DynaQAgent): the learning agent
        Episodes (int): number of episodes per iteration
        Repetitions (int): the number of repetitions for the experiment
    Returns:
        Average number of steps per episode (np.array): has shape of (repetitions, episodes)
        and the value at index i, j is the number of steps, for iteration number i, for the
        j-th episode to finish.
    """
    repetitions = experiment_args['repetitions']
    episodes = experiment_args['episodes']

    avg_steps_per_episode = np.zeros((repetitions, episodes))
    for rep in tqdm(range(repetitions)):
        env = DynaMaze(env_args)
        env.add_obstacles([(i, 2) for i in range(1, 4)] + [(i, 7) for i in range(0, 3)] + [(4,5)])
        agent_args['seed'] = rep
        agent = DynaQAgent(env, agent_args)

        for ep in range(episodes):
            # Reset the environment (move agent to the start)
            # Gymnasium requires this step is called at the start of each episode
            s = env.reset()
            terminated = False
            curr_steps = 0

            while not terminated:
                s, terminated, _ = agent.step(s)
                curr_steps += 1
            avg_steps_per_episode[rep][ep] = curr_steps

    # [1:] because the first episode is the same for all values of n
    return np.mean(avg_steps_per_episode, axis=0)[1:]

if __name__ == '__main__':
    # The environment, values, etc. are taken from the example 8.1 in the intro to RL book (Barto & Sutton)
    n_values = [0, 5, 50]

    experiment_args = {'episodes': 50, 'repetitions': 30}
    num_of_steps = np.zeros((len(n_values), experiment_args['episodes']-1))
    agent_args = {'n': 0, 'gamma': 0.95, 'alpha': 0.1, 'epsilon': 0.1}
    env_args = {'iter_change': 100000000, 'start_state': (2, 0), 'goal_state': (0, 8)}

    for i, n in enumerate(n_values):
        agent_args['n'] = n
        num_of_steps[i] = experiment(env_args, agent_args, experiment_args)

    plot_results(
        np.arange(2, experiment_args['episodes'] + 1),
        num_of_steps,
        'Episodes',
        'Steps per Episode',
        [f'n={n}' for n in n_values],
        './libs/graphs/results/dynaQ_maze.png')