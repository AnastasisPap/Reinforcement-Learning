import sys, os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

import numpy as np
from libs.envs.dyna_maze import DynaMaze
from libs.graphs.graphing import plot_results
from planning_and_learning.model import Model
from tqdm import tqdm

class DynaQAgent:
    """
    Args:
        Env: object of DynaMaze which is the agent environment
        args: dictionary mapping name (str) -> value (int or float)
    """
    def __init__(self, env, args):
        self.epsilon = args['epsilon']
        self.gamma = args['gamma']
        self.alpha = args['alpha']
        self.n = args['n']
        self.env = env
    
    """Resets variables of the agent. This is called each time we do a new
    run.

    Args:
        The random seed (int) which will be used for the epsilon-greedy and
        sampling from the model
    """
    def reset(self, seed=42):
        self.model = Model(env, seed)
        self.rnd_gen = np.random.RandomState(seed)
        self.Q = np.zeros((self.env.height, self.env.width, self.env.action_space.n))
    
    """It's the epsilon greedy policy. With probability epsilon selects any action
    randomly with the same probability, and with prob 1-epsilon, greedily selects
    the action and breaks ties arbitrarily.

    Args:
        The state (tuple of ints) for which the policy will be used.
    Returns:
        The action (int) which is an int in [0, 3]) which the policy chose.
    """
    def eps_greedy(self, s):
        prob = self.rnd_gen.uniform()
        if prob < self.epsilon:
            return self.rnd_gen.randint(self.env.action_space.n)
        else:
            return self.rnd_gen.choice(np.where(np.max(self.Q[s]) == self.Q[s])[0])
    
    """For Dyna-Q agents with n>0, the following is the planning step.
    """
    def planning(self):
        for i in range(self.n):
            s, a = self.model.sample()
            next_s, r = self.model.get(s, a)
            self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])

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
    def step(self, s):
        a = self.eps_greedy(s)

        next_s, r, is_terminal = self.env.step(a)

        self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s][a])
        self.model.update(s, a, r, next_s)

        self.planning()
        
        return next_s, is_terminal
    
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
def experiment(agent, episodes=50, repetitions=30):
    avg_steps_per_episode = np.zeros((repetitions, episodes))
    for rep in tqdm(range(repetitions)):
        # Reset the agent (resets Q values, uses new Random Generator)
        agent.reset(rep)
        for ep in range(episodes):
            # Reset the environment (move agent to the start)
            # Gymnasium requires this step is called at the start of each episode
            s = agent.env.reset()
            terminated = False
            curr_steps = 0

            while not terminated:
                curr_steps += 1
                s, terminated = agent.step(s)
            avg_steps_per_episode[rep][ep] = curr_steps

    # [1:] because the first episode is the same for all values of n
    return np.mean(avg_steps_per_episode, axis=0)[1:]

if __name__ == '__main__':
    # The environment, values, etc. are taken from the example 8.1 in the intro to RL book (Barto & Sutton)
    env = DynaMaze(start_state=(2, 0), goal_state=(0, 8))
    env.add_wall((1, 2), (3, 2))
    env.add_wall((4, 5), (4, 5))
    env.add_wall((0, 7), (2, 7))
    n_values = [0, 5, 50]
    episodes = 50
    repetitions = 30
    num_of_steps = np.zeros((len(n_values), episodes-1))
    args = {'n': 0, 'gamma': 0.95, 'alpha': 0.1, 'epsilon': 0.1}

    for i, n in enumerate(n_values):
        args['n'] = n
        agent = DynaQAgent(env, args)
        num_of_steps[i] = experiment(agent, episodes, repetitions)

    plot_results(
        np.arange(2, episodes + 1),
        num_of_steps,
        'Episodes',
        'Steps per Episode',
        [f'n={n}' for n in n_values],
        './libs/graphs/results/dynaQ_maze.png')