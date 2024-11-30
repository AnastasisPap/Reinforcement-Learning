<br />
<div align="center">
  <h3 align="center">Reinforcement Learning Experiments</h3>
</div>

<!-- ABOUT THE PROJECT -->
## About

This repo reproduces mostly all experiments done in the book "Introduction to Reinforcement Learning" by Barto & Sutton, (sidenote: this is not the official code). It has been built in a way that an experiment can be done in a "plug and play" fashion, making it very easy to use any of the supported RL algorithms and Environments to create an experiment. Collecting statistics and making plots and graphs is also very easy to do.

The experiments cover the following topics taken by the book:
* Multi-armed Bandits
* Dynamic Programming/Markov Decision Processes
* Monte Carlo methods
* Temporal Difference learning
* N-step bootstrapping methods
* Planning and learning methods

## Getting Started

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Python >=3.8 (tested on 3.9.15)
* Libraries
    * Numpy
    * Matplotlib
    * Gymnasium
    * Scipy
    * Tqdm
    * Pygame
    * Seaborn

<u><i>Suggestion:</i></u> use a conda environment to avoid possible package errors.

### Installation
* PIP:
   ```sh
   pip install -r ./path/to/requirements.txt
   ```
* <b>Important</b>
If errors like "libs.envs.blackjack.py is not found" appear, enter the following command in the CLI/Terminal:
   * For Linux/Mac
   ```sh
   export PYTHONPATH=$(pwd)
   ```
    * For Windows follow the following [tutorial](https://realpython.com/add-python-to-path/).


## Environments
Each environment must have some required methods depending on the type of agent which will be used on. Specifically there are three types: environments for armed bandits, MDP agents and learning agents. Each environment takes in a python dictionary when initialized.

### Armed bandits Environments:
Available environments:
* [Armed testbed](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/armed_testbed.py)

Environment args:
| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
| `n`  | `int`|  The number of arms.                     | `10` |
| `mean`  | `float`|  The mean which will be used to generate the q* function. | `0` |
| `var`  | `float`|  The variance which will be used to generate the q* function and the reward. | `1` |

<br>

To make a custom environment for armed bandits, the env must have the following methods:
* `reset(seed)`: called at the start of each experiment.
* `step(action)`: called at each iteration, takes in an action and returns the reward.

### MDP Environments:
Each MDP environment MUST have known dynamcis (transition probabilities) and known state space.

Available environments:
* [Car rental](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/car_rental.py)
Environment args:

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`lambda_first`|`tuple(int,int)`| The lambdas used as the Poisson parameters for the first location. The first item is the lambda for requests and the second for returns|`(3,3)`|
|`lambda_second`|`tuple(int,int)`| The lambdas used as the Poisson parameters for the second location. The first item is the lambda for requests and the second for returns|`(4,2)`|
|`max_cars`|`int`|The maximum number of cars that can be held in each location.|`20`|
|`max_cars_moved`|`int`|The maximum number of cars that can be moved in a day.|`5`|
|`credit`|`int`|The credit given for each car that is rented.|`10`|
|`cost`|`int`|The cost for moving a car from one location to the other.|`2`|
|`min_prob`|`float`|To avoid extra computation, probabilities from Poisson sampling lower than the min_prob will not be considered.|`1e-6`|

* [Coin flip gamble](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/coin_flip_gamble.py)
Environment args:

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`max_capital`|`int`| The capital that the agent aims to reach (i.e. reaching this capital will end the game).|`100`|
|`p`|`float in [0,1]`| The probability of winning the bet.|`0.4`|

<br>

To make a custom environment for MDP agents, the env must have the following methods:
* `get_actions(state)`: returns all the possible actions the agent can take if they currently are at the given state.
* `check_state_action(s,a)`: returns True if the given pair is possible to happen (for example (99,10) isn't possible for the gambler's problem), otherwise returns False.
* `get_dynamics()`: returns a dictionary that maps to another dictionary which maps to a probability. The outer dictionary contains (state,action) pairs and the inner contains (state', reward) pairs. For example [(s,a)][(s',r)] = p means that if we currently are at state s and take action a, the probability of transitioning to state s' and taking reward r is equal to p.
* `observation_space`: it's a list that contains all the possible valid states.

### Learning Environments
<p align="right">(<a href="#readme-top">back to top</a>)</p>

Available Environments:
* [Blackjack](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/blackjack.py)

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`natural`|`bool`| Whether to give an additional reward for starting with a natural blackjack (taken from gymlibrary documentation).|`False`|
|`sab`|`boo`| Whether to follow the exact rules outlined in the book by Sutton and Barto (taken from gymlibrary documentation).|`False`|

* [Random Walk](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/random_walk.py)

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`n`|`int`| Number of states|`5`|
|`left_r`|`float`| Reward given when reaching the leftmost state.|`0`|
|`right_r`|`float`| Reward given when reaching the rightmost state.|`1`|

* [Racetrack](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/racetrack.py)

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`dimensions`|`tuple(int,int)`| The dimensions of the racetrack as a rectangle.|`(32,17)`|
|`size`|`int`| Used to scale the grid squares when rendering the game using pygame.|`20`|
|`render_mode`|`'human' or None`| If human then the game is rendered using pygame.|`None`|
|`agent_marker`|`int`| In the grid array, this is used to mark the location of the agent.|`4`|
|`start_states`|`list[tuple(int,int)]`| Contains the locations of all the start states.|`[(31,3),(31,4),(31,5),(31,6),(31,7),(31,8)]`|
|`goal_states`|`list[tuple(int,int)]`| Contains the locations of all the goal states.|`[(0,16),(1,16),(2,16),(3,16),(4,16),(5,16)]`|
|`boundaries`|`list[tuple(int,int)]`| Contains the locations of all the boundaries (i.e. if the agent steps on these, the game resets).|omitted|

* [Grid World](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/grid_world.py)

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`dimensions`|`tuple(int,int)`| The dimensions of the grid world.|`(6,9)`|
|`start_state`|`tuple(int,int)`| The location of the start state.|`(2,0)`|
|`goal_state`|`tuple(int,int)`| The location of the goal state.|`(0,8)`|

* [Cliff Walk](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/cliff_walk.py)

**Along with the same parameters as the Grid World Environment, it contains:**

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`cliff`|`list[tuple(int,int)]`| The locations of the cliff (i.e. the game restarts if the agent steps on these).|omitted|
|`seed`|`int`| Seed used for reproducibility.|`0`|

* [Windy Grid World](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/windy_grid_world.py)

**Along with the same parameters as the Grid World Environment, it contains:**

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`Wind strength`|`list[int]`| The wind strength for each column of the grid world.|omitted|

* [Dyna Maze](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/envs/dyna_maze.py)

**Along with the same parameters as the Grid World Environment, it contains:**

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`iter_change`|`int`| The iteration in which the obstacle change happens (if no changes happens, set it at infinity).|`10000`|
|`obstacles`|`list[tuple(int,int)]`| The locations of all obstacles.|`[ ]`|
|`obstacles_after_change`|`list[tuple(int,int)]`| The locations of all obstacles AFTER the obstacle change happens.|`[ ]`|

To create a custom learning environment the following methods must be present:
* `env_dimensions` which is used to initialize the shape of the state-value, state-action functions.
* `action_space` which is the space of all possible actions.
* `reset(seed)` which is called at the start of each episode.
* `is_terminal(s)` which returns True if s is a terminal state.
* `step(a)` takes action a and returns: (next state, reward, if next state is terminal or not)

### Agents
For each learning method, there can be multiple agent algorithms. Specifically:
* Multi-armed Bandits
  * [Simple bandit](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/multi_armed_bandits/bandit.py)
  * [UCB](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/multi_armed_bandits/bandit.py)
  * [Gradient Bandit](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/multi_armed_bandits/gradient_bandit.py)
* Dynamic Programming
  * [Policy Iteration](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/dynamic_programming/mdp_policy_iteration.py)
  * [Value Iteration](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/dynamic_programming/value_iteration.py)
* Monte Carlo
  * [First Visit Prediction](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/monte_carlo/first_visit_prediction.py)
  * [Exploring Starts](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/monte_carlo/exploring_starts.py)
  * [Off-policy Monte Carlo Control](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/monte_carlo/off_policy_mc_control.py)
* Temporal Difference learning
  * [TD(0)](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/temporal_difference_learning/td_zero.py)
  * [Sarsa](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/temporal_difference_learning/sarsa.py)
  * [Q-Learning](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/temporal_difference_learning/q_learning.py)
* N-step Bootstrapping
  * [N-step TD](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/n_step_bootstrapping/n_step_td.py)
* Planning and Learning methods
  * [Dyna-Q](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/planning_and_learning/dynaQ.py)
  * [Dyna-Q+](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/planning_and_learning/dynaQPlus.py)
  * [Prioritized Sweeping](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/planning_and_learning/prioritized_sweeping.py)

To create a new <b>bandit algorithm</b> the class must contain:
* `step()` which takes a step and returns a tuple (action, reward) which is the action it selected and the reward it got by taking the action.
  
To create a new <b>MDP agent</b> the class must:
* Inherit from `libs.utils.agent.BaseAgent` and pass the env and args during initialization.
* Contain the method `iteration()`, which returns True iff the policy is stable.
* Contains `policy` as a variable which is a dictionary mapping states to actions.
* Contains `sweeps` list which store the state value function at each sweep.

To create a new <b>general learning agent</b> the class must:
* Inherit from `libs.utils.agent.BaseAgent` and pass the env and args during initialization.
* Must have a `step(s)` function which takes in a state and produces a triplet (next state, is next state terminal, reward).
* For custom parameters that are abscent from BaseAgent, they can be set by using the args dictionary.

[BaseAgent](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/utils/agent.py) args:

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`epsilon`|`float`| Agents that use ε-greedy policies.|`0.1`|
|`gamma`|`float`| The discount.|`1.0`|
|`alpha`|`float`| The learning rate.|`0.1`|
|`init_value`|`float or 'normal' or 'uniform'`| If it's a float, then the state-value and state-action functinos are initialized with this value. If it's uniform then they are initialized using the uniform distribution and if it's normal then they are initialized using the normal distribution.|`0.0`|
|`dtype`|`type`| The dtype of the state-value and state-action functions.|`float`|


Any agent that inherits `BaseAgent` can override its methods.

### Experiments
There is an experiment method for each type of agent there is, so there is one for bandits, MDP, general learning agents. Each experiment function works in a similar way. They take in the Environment Class, Agent Class, and experiment, environment, agent args and they return a data object.

Currently they store a variety of statistics which are measured in between episodes and iterations. They initialize the environment, agent and use the step function of the agent to finish the episode.

When deciding which function to use if the agent you want to use isn't a bandit or MDP, then it's the general experiment.

[General Experiment](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/utils/experiment.py)
Experiment args:

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`repetitions`|`int`| The number of repetitions to run an experiment (the agent will be completely initialized each time).|`1`|
|`episodes`|`int`| The number of episodes for each repetition.|`10000000`|
|`max_steps`|`int`| The maximum number of agent steps for each repetition.|`10000000`|
|`true_values`|`list[float]`| The true values of the state-action/state-value functions. This is optional and only needed if we need to compare the learned Q and the real Q.|None|

Experiment Statistics:

| Name | Description |
|----------------------|------------------------------------------------------------------------|
|`avg_steps_per_episode`| The average (average in the number of repeitions) number of steps it took to finish an episode.|
|`cum_reward`| The average (average in the number of repetitions) cumulative reward for each repetition.|
|`episodes_per_time_step`|The average (average in the number of repetitions) number of episodes before reaching max iterations.|
|`cum_reward_per_episode`|The average (average in the number of repetitions) cumulative reward within each episode.|
|`rms_error`|The average (average in the number of repetitions) Root Mean Squared Error per episode.|
|`policy_per_rep`|Stores the policy of the agent for each repetition|
|`estimated_values_per_episode`|Stores the average (average in the number of repetitions) value function per episode.|


[DP Experiment](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/utils/experiment.py)
Experiment Args

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`update_freq`|`int`| The frequency at which a print statement will happen to update the user.|`1`|
|`store_data`|`boo`| If true then the result of the experiment is stored as a pickle file in the given data path.|`False`|
|`data_path`|`str`| The path at which the data will be stored at.|Required if store_data is True|

Experiment Statistics:

| Name | Description |
|----------------------|------------------------------------------------------------------------|
|`value`| The value function at the last iteration|
|`policies`| The policy of the agent at each iteration (until the policy becomes stable).|
|`sweeps`|The value function at each sweep.|

[Bandit Experiment](https://github.com/AnastasisPap/Reinforcement-Learning/blob/main/libs/utils/experiment.py)
Experiment Args

| Parameter     | Type  | Description                                             | Default Value |
|---------------|-------|---------------------------------------------------------|---------------|
|`Runs`|`int`| The number of runs.|`1000`|
|`iterations`|`int`| The number of iterations (for each iteration there are # runs).|`2000`|

Experiment Statistics:

| Name | Description |
|----------------------|------------------------------------------------------------------------|
|`avg_rewards`| The average rewards (average in the number of iterations).|
|`chosen_opt`| Average number of times the agent chose the optimal value (average in the numbe of iterations).|

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Anastasios Papapanagiotou - anastasis.papapanagiotou@gmail.com

Project Link: [https://github.com/AnastasisPap/Reinforcement-Learning](https://github.com/your_username/repo_name)

## Sources
* MD template taken from this [repo](https://github.com/othneildrew/Best-README-Template/blob/main/README.md)
* [Reinforcement learning: An Introduction, 2nd edition](https://www.academia.edu/download/38529120/9780262257053_index.pdf)
