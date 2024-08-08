import numpy as np
from libs.utils.agent import BaseAgent
from planning_and_learning.model import Model

class DynaQAgent(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.n = args.get('n', 10)
        self.rnd_gen = np.random.RandomState(args.get('seed', 0))
        self.model = Model(env, self.rnd_gen)

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