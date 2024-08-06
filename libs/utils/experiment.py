import numpy as np
from tqdm import tqdm

def experiment(EnvClass, AgentClass, env_args, agent_args, experiment_args):
    """Completes the experiment. To properly calculate the number of steps required to complete
    the episode, multiple iterations are used and then averaged. For each iteration, multiple
    episodes are used. At the start of each new run, the agent is reset, while the env is
    reset every new episode.

    Args:
        EnvClass (gymnasium.Env): the environment class
        AgentClass (BaseAgent): the learning agent class
        env_args (dict): the arguments for the environment
        agent_args (dict): the arguments for the agent
        experiment_args (dict): the arguments for the experiment
    Returns:
        Data (dict): the data collected during the experiment, currently only the average steps
        per episode and the cumulative reward
    """
    repetitions = experiment_args.get('repetitions', 1)
    episodes = experiment_args.get('episodes', 100000)
    max_steps = experiment_args.get('max_steps', 100000)
    data = {}
    data['avg_steps_per_episode'] = np.zeros((repetitions, episodes))
    data['cum_reward'] = np.zeros((repetitions, max_steps))

    for rep in tqdm(range(repetitions)):
        env = EnvClass(env_args)
        agent_args['seed'] = rep
        agent = AgentClass(env, agent_args)

        total_steps = 0
        total_reward = 0
        total_episodes = 0

        while total_episodes < episodes and total_steps < max_steps - 1:
            # Reset the environment (move agent to the start)
            # Gymnasium requires this step is called at the start of each episode
            s = env.reset()
            agent.reset(s)
            terminated = False
            curr_steps = 0

            while not terminated and total_steps < max_steps - 1:
                s, terminated, r = agent.step(s)

                curr_steps += 1
                total_steps += 1
                total_reward += r

                data['cum_reward'][rep][total_steps] = total_reward

            data['avg_steps_per_episode'][rep][total_episodes] = curr_steps
            total_episodes += 1

    return data