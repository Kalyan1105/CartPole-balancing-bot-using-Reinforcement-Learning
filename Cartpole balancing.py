import gym
import numpy as np
import random
import os

# Set up the CartPole environment
env = gym.make('CartPole-v1')

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration probability
episodes = 1000     # Number of training episodes
max_steps = 200     # Max steps per episode
q_table_file = 'q_table.npy'  # File to save/load Q-table

# Discretization settings
n_buckets = (6, 6, 12, 12)  # Number of buckets per state dimension (cart position, cart velocity, pole angle, pole angular velocity)
state_bounds = [
    (-4.8, 4.8),           # Cart position
    (-0.5, 0.5),           # Cart velocity (custom bound)
    (-0.418, 0.418),       # Pole angle (~24 degrees)
    (-0.872, 0.872)        # Pole angular velocity (~50 degrees in radians)
]

# Initialize Q-table
q_table_shape = n_buckets + (env.action_space.n,)
if os.path.exists(q_table_file):
    q_table = np.load(q_table_file)
    print(f"Loaded Q-table from {q_table_file}")
else:
    q_table = np.zeros(q_table_shape)
    print("Initialized new Q-table")

# Discretize the state space
def discretize_state(state):
    try:
        # Clip state to ensure it stays within bounds
        state = np.clip(state, [b[0] for b in state_bounds], [b[1] for b in state_bounds])
        # Normalize state to [0, 1]
        state_adj = (state - np.array([b[0] for b in state_bounds])) / (
            np.array([b[1] for b in state_bounds]) - np.array([b[0] for b in state_bounds])
        )
        # Discretize each dimension
        discretized = [
            min(max(int(np.digitize(s, np.linspace(0, 1, n_buckets[i])) - 1, 0), n_buckets[i] - 1)
            for i, s in enumerate(state_adj)
        ]
        return tuple(discretized)
    except Exception as e:
        print(f"Error discretizing state: {e}")
        return tuple([0] * len(n_buckets))  # Return default state on error

# Training loop
rewards = []
for episode in range(episodes):
    try:
        state = discretize_state(env.reset()[0])  # Handle new Gym versions
        done = False
        total_reward = 0

        for step in range(max_steps):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Perform action
            try:
                obs, reward, done, _, _ = env.step(action)  # Handle new Gym step return
            except Exception as e:
                print(f"Error during step: {e}")
                break

            new_state = discretize_state(obs)

            # Update Q-table
            best_future_q = np.max(q_table[new_state])
            q_table[state + (action,)] += alpha * (
                reward + gamma * best_future_q - q_table[state + (action,)]
            )

            state = new_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Output progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if rewards else 0
            print(f"Episode {episode}, Total Reward: {total_reward}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    except Exception as e:
        print(f"Error in episode {episode}: {e}")
        continue

# Save Q-table
np.save(q_table_file, q_table)
print(f"Saved Q-table to {q_table_file}")

# Test the trained agent
def test_agent(episodes=10):
    total_rewards = []
    for episode in range(episodes):
        try:
            state = discretize_state(env.reset()[0])
            done = False
            total_reward = 0
            for _ in range(max_steps):
                action = np.argmax(q_table[state])
                obs, reward, done, _, _ = env.step(action)
                state = discretize_state(obs)
                total_reward += reward
                if done:
                    break
            total_rewards.append(total_reward)
            print(f"Test Episode {episode}, Total Reward: {total_reward}")
        except Exception as e:
            print(f"Error in test episode {episode}: {e}")
            continue
    avg_test_reward = np.mean(total_rewards) if total_rewards else 0
    print(f"Average test reward over {episodes} episodes: {avg_test_reward:.2f}")

# Run test
test_agent()

# Close environment
env.close()