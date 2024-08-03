import numpy as np
from tqdm import tqdm

def experiment(EnvClass, AgentClass, env_args, agent_args, experiment_args):
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

        while total_episodes < episodes - 1 and total_steps < max_steps - 1:
            # Reset the environment (move agent to the start)
            # Gymnasium requires this step is called at the start of each episode
            s = env.reset()
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