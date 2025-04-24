# CartPole-balancing-bot-using-Reinforcement-Learning
Q-Learning for CartPole-v1
This project implements a Q-learning algorithm to train an agent to solve the CartPole-v1 environment from OpenAI Gym. The agent learns to balance a pole on a cart by discretizing the continuous state space and updating a Q-table using an epsilon-greedy policy.
Overview
The code performs the following tasks:

Environment Setup: Initializes the CartPole-v1 environment.
State Discretization: Converts the continuous state space into discrete buckets for Q-learning.
Q-Learning: Trains the agent using the Q-learning algorithm with an epsilon-greedy action selection strategy.
Training Loop: Runs multiple episodes, updating the Q-table based on rewards and decaying exploration over time.
Progress Monitoring: Prints total rewards every 100 episodes.

Requirements
To run this code, ensure you have the following dependencies installed:

Python 3.7+
Gym (OpenAI)
NumPy

You can install the dependencies using:
pip install gym numpy

Note: This code is compatible with Gym versions < 0.26.0. For newer versions, minor modifications are required (see Notes).
Usage

Install Dependencies: Ensure Gym and NumPy are installed.
Run the Script: Execute the Python script in a compatible environment (e.g., Jupyter Notebook, Python console, or IDE).python cartpole_q_learning.py


Output: The script will:
Train the agent for 1000 episodes.
Print the total reward every 100 episodes.
Store the learned Q-table in memory (not saved to disk by default).



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
n_buckets = (6, 6, 12, 12): Number of buckets for each state dimension (cart position, cart velocity, pole angle, pole angular velocity).
Custom bounds for velocity ([-0.5, 0.5]) and angular velocity ([-50°, 50°]).



Code Structure

Environment: Uses gym.make('CartPole-v1') to create the CartPole environment.
Discretization: The discretize_state function maps continuous states to discrete buckets.
Q-Learning:
Initializes a Q-table with zeros.
Updates the Q-table using the Q-learning formula.
Uses epsilon-greedy action selection.


Training Loop: Iterates over episodes, performs actions, updates the Q-table, and decays epsilon.

Notes

Gym Compatibility: The code assumes env.reset() returns the observation directly, which is true for Gym < 0.26.0. For newer versions, modify the reset call:current_state = discretize_state(env.reset()[0])


Discretization Robustness: The discretization function may produce out-of-bounds indices for edge cases. Consider clipping states or adding boundary checks.
Performance Monitoring: The code only prints total rewards every 100 episodes. For better monitoring, track average rewards or success rates.
Saving the Q-Table: The Q-table is not saved by default. Add np.save('q_table.npy', q_table) to save it for reuse.
Testing: The code does not include a testing phase. Add a function to evaluate the trained policy without exploration (see suggested improvements).

Suggested Improvements

Handle Newer Gym Versions: Update env.reset() and env.step() to handle tuple returns.
Robust Discretization: Add clipping to ensure states stay within bounds:state = np.clip(state, env.observation_space.low, env.observation_space.high)


Enhanced Monitoring: Track and print average rewards over the last 100 episodes.
Testing Phase: Add a function to test the trained agent:def test_agent(episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = discretize_state(env.reset()[0])
        done = False
        total_reward = 0
        for _ in range(max_steps):
            action = np.argmax(q_table[state])
            obs, reward, done, _ = env.step(action)
            state = discretize_state(obs)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    print(f"Average test reward: {np.mean(total_rewards):.2f}")


Save Q-Table: Save the Q-table to disk for reuse.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using OpenAI Gym for the CartPole environment.
Implements the Q-learning algorithm for reinforcement learning.

