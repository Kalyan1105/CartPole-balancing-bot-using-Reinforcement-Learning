# CartPole-balancing-bot-using-Reinforcement-Learning
Q-Learning for CartPole-v1
This project implements a Q-learning algorithm to train an agent to solve the CartPole-v1 environment from OpenAI Gym. The agent learns to balance a pole on a cart by discretizing the continuous state space and updating a Q-table using an epsilon-greedy policy. The code includes robust state handling, performance monitoring, Q-table persistence, and a testing phase.
Overview
The code performs the following tasks:

Environment Setup: Initializes the CartPole-v1 environment.
State Discretization: Converts the continuous state space into discrete buckets for Q-learning.
Q-Learning: Trains the agent using the Q-learning algorithm with an epsilon-greedy action selection strategy.
Training Loop: Runs multiple episodes, updates the Q-table, and decays exploration over time.
Performance Monitoring: Tracks and prints total and average rewards.
Q-Table Persistence: Saves and loads the Q-table to/from a file.
Testing: Evaluates the trained policy over multiple episodes without exploration.

Requirements
To run this code, ensure you have the following dependencies installed:

Python 3.7+
Gym (OpenAI)
NumPy

You can install the dependencies using:
pip install gym numpy

Note: The code is compatible with Gym versions >= 0.26.0 (new API) and older versions.
Usage

Install Dependencies: Ensure Gym and NumPy are installed.
Run the Script: Execute the Python script in a compatible environment (e.g., Jupyter Notebook, Python console, or IDE).python cartpole_q_learning.py


Output: The script will:
Train the agent for 1000 episodes.
Print total reward, average reward (last 100 episodes), and epsilon every 100 episodes.
Save the Q-table to q_table.npy.
Test the trained agent for 10 episodes and print the average test reward.
Example output:Episode 0, Total Reward: 23.0, Avg Reward (last 100): 23.00, Epsilon: 1.000
Episode 100, Total Reward: 45.0, Avg Reward (last 100): 38.50, Epsilon: 0.605
...
Test Episode 0, Total Reward: 195.0
Average test reward over 10 episodes: 190.50





Key Parameters

Hyperparameters:
alpha = 0.1: Learning rate for Q-table updates.
gamma = 0.99: Discount factor for future rewards.
epsilon = 1.0: Initial exploration rate.
epsilon_min = 0.01: Minimum exploration rate.
epsilon_decay = 0.995: Decay rate for exploration probability.
episodes = 1000: Number of training episodes.
max_steps = 200: Maximum steps per episode.


Discretization:
n_buckets = (6, 6, 12, 12): Number of buckets for cart position, cart velocity, pole angle, and pole angular velocity.
state_bounds: Custom bounds for each state dimension:
Cart position: ±4.8
Cart velocity: ±0.5
Pole angle: ±24° (±0.418 radians)
Pole angular velocity: ±50° (±0.872 radians).




Q-Table File: q_table.npy for saving/loading the Q-table.

Code Structure

Environment: Uses gym.make('CartPole-v1') to create the CartPole environment.
Discretization: The discretize_state function maps continuous states to discrete buckets with clipping and error handling.
Q-Learning:
Initializes or loads a Q-table.
Updates the Q-table using the Q-learning formula.
Uses epsilon-greedy action selection.


Training Loop: Iterates over episodes, performs actions, updates the Q-table, and decays epsilon.
Testing: Evaluates the trained policy without exploration.
Persistence: Saves the Q-table to disk for reuse.

Notes

Gym Compatibility: The code supports Gym >= 0.26.0 by handling tuple returns from env.reset() and env.step().
Robustness: Includes error handling for state discretization, environment steps, and episode execution.
Discretization: States are clipped to prevent out-of-bounds errors, and indices are constrained to valid ranges.
Performance Monitoring: Tracks rewards and prints average performance over the last 100 episodes.
Q-Table Persistence: The Q-table is saved to q_table.npy and loaded if available, allowing training to resume.
Testing: The test_agent function evaluates the policy over 10 episodes, providing a clear measure of performance.

Suggested Improvements

Visualization: Add rendering of the environment during testing to visualize the agent's behavior:env = gym.make('CartPole-v1', render_mode='human')


Hyperparameter Tuning: Experiment with different alpha, gamma, epsilon_decay, or n_buckets values.
Reward Smoothing: Use a moving average or exponential smoothing for more stable reward tracking.
Early Stopping: Stop training if the average reward stabilizes (e.g., consistently near 200).
Logging: Save training logs to a file for analysis.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using OpenAI Gym for the CartPole environment.
Implements the Q-learning algorithm for reinforcement learning.


