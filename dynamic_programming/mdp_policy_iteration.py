from libs.utils.agent import BaseAgent

class MDPPolicyIter(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.theta = args.get('theta', 1e-5)

        # The dynamics must be known for MDP algorithms
        assert hasattr(env, 'get_dynamics'), 'MDPs require environment dynamics to be known.'
        assert hasattr(env, 'observation_space'), 'MDPs require environment observation space to be known.'
        self.observation_space = env.observation_space

        load = args.get('load_dynamics', False) # If set to True then the dynamics will be loaded from a file
        # If load is False, then the dynamics will be generated and stored in this path
        path = args.get('dynamics_path', './libs/envs/dynamics.pkl') 
        self.dynamics = env.get_dynamics(path, load)
        self.policy = {s: 0 for s in self.observation_space}
    
    def iteration(self):
        self.policy_evaluation()
        return self.policy_improvement()
        
    def policy_evaluation(self):
        delta = float('inf')
        self.sweeps = []

        while delta >= self.theta:
            delta = 0

            for s in self.observation_space:
                prev_value = self.V[s]

                new_v = 0
                a = self.policy[s]
                
                for (s_prime, r), prob in self.dynamics[s, a].items():
                    new_v += prob * (r + self.gamma * self.V[s_prime])
                self.V[s] = new_v
                delta = max(delta, abs(prev_value - self.V[s]))
            self.sweeps.append(self.V.copy())
            
        return delta
    
    def policy_improvement(self):
        policy_stable = True

        for s in self.observation_space:
            old_action = self.policy[s]
            max_v = -1

            for a in self.env.get_actions(s):
                if not self.env.check_state_action(s, a): continue

                v = sum(
                    [
                        prob * (r + self.gamma * self.V[s_prime])
                        for (s_prime, r), prob in self.dynamics[s, a].items()
                    ])
                
                if v >= max_v:
                    max_v = v
                    self.policy[s] = a

            if old_action != self.policy[s]:
                policy_stable = False
        
        return policy_stable