import gymnasium
import numpy as np
from multi_armed_bandit.env import MultiArmedBandit

# Create the environment
env = gymnasium.make('MultiArmedBandit-v0', render_mode=None, num_actions=10, samples=1000, show_true_distributions=True)
env.set_testbed()
env.reset()

# Setting up the e-greedy algorithm parameters
steps = 10
num_actions = env.action_space.n
Q_t = np.zeros((num_actions, 3), dtype=np.half)  # sum, count, mean

# Initialize the mean values of the reward distributions with random values
random_numbers = np.random.rand(10, 3)
Q_t[:, -1] = np.random.default_rng().uniform(0, 1, size=(10, 3))[:, -1]

epsilon = 0.1
for i in range(steps):

    # Select the action with the highest mean with probability epsilon
    action = np.argmax(Q_t[:, 2])  # Selecting the action with the highest mean value

    if np.random.default_rng().uniform(0, 1) <= epsilon:
        action = np.random.default_rng().integers(0, num_actions)

    observation, reward, terminated, truncated, info = env.step(action)  # Execute the selected action

    # Update the Q_t table
    Q_t[action, 0] += reward
    Q_t[action, 1] += 1
    Q_t[action, 2] = Q_t[action, 0] / Q_t[action, 1]

    print(Q_t)
    print()
