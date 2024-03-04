import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc


class MultiArmedBandit(gym.Env):
    """A simple multi-armed bandit environment.
    """

    metadata = {"render_modes": [None, 'human', 'interactive_plot']}

    def __init__(self, render_mode: str = None, num_actions: int = 10, mean_interval: tuple[int, int] = (-3, 3),
                 scale: int = 10, samples: int = 100, show_true_distributions: bool = True):
        """Initialize the environment.

        Args:
            render_mode: (str) The mode in which the environment should be rendered (e.g. "human", "interactive_plot" or None).
            num_actions: (int) The number of actions (levers in case of the slot machine example).
            mean_interval: (tuple) The interval from which the mean of the reward distribution is sampled.
            scale: (int) Scale or "width" the reward distribution.
            samples: (int) The number of samples in each reward distribution.
            show_true_distributions: (bool) Whether to show the real reward distributions or not.
        """

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode  # The mode in which the environment should be rendered (e.g. "human" or None)

        self.num_actions = num_actions  # The number of actions (levers in case of the slot machine example)
        self.mean_interval = mean_interval  # The interval from which the mean of the reward distribution is sampled
        self.scale = scale  # Scale or "width" the reward distribution
        self.samples = samples  # The number of samples in each reward distribution
        self.show_true_distributions = show_true_distributions  # Whether to show the real reward distributions or not

        self.reward_distributions = None
        self._empirical_rewards = None
        self._cum_reward = None
        self._rew_list = None
        self._history = None

        self.observation_space = spaces.Discrete(1, start=0)  # Observations are empty
        self.action_space = spaces.Discrete(num_actions)

    def _get_obs(self):
        return 0

    def _get_info(self):
        return {}

    def set_testbed(self, testbed: npt.ArrayLike = None, seed=None):

        # TODO: Streamline setting the testbed

        reward_distributions = testbed
        if reward_distributions is None:
            print("Setting random the testbed...")
            # Generate the means of the reward distributions
            means = np.random.default_rng().normal(loc=np.random.randint(*self.mean_interval), scale=self.scale,
                                                   size=self.num_actions)

            # Generate a reward distribution for each action
            reward_distributions = np.random.default_rng().normal(loc=means[:, None], scale=self.scale,
                                                                  size=(self.num_actions, self.samples))

            self.reward_distributions = reward_distributions

            # return reward_distributions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO: Call reset automatically when creating the env

        if self.reward_distributions is None:
            raise ValueError("The testbed is not set. Please call the env.set_testbed() method first.")

        observation = self._get_obs()
        info = self._get_info()

        self._empirical_rewards = np.zeros((self.num_actions, 1)).tolist()
        self._cum_reward = 0
        self._rew_list = [0]
        self._history = []

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Sample randomly from the reward distribution for the selected action
        reward = self.reward_distributions[action][np.random.randint(0, self.samples)]
        observation = self._get_obs()
        info = self._get_info()

        # Track the empirical rewards
        self._cum_reward += reward
        self._rew_list.append(self._cum_reward)
        self._empirical_rewards[action].append(reward)
        self._history.append((action, reward))

        # Check if the episode is terminated
        terminated = False

        if self.render_mode == "human":
            self._render_frame(action)

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def _render_frame(self, action="-"):
        if action != "-":
            action += 1

        self.fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

        cum_reward_formatted = "{:.2f}".format(self._cum_reward)
        self.fig.suptitle(f'Multi-armed bandit\nCumulative reward: {cum_reward_formatted}', fontsize=14)

        # Plot the reward distributions
        ax1.set_xlabel(f'Action ({action})', fontsize=12)
        ax1.set_ylabel('Reward\ndistribution', fontsize=12)
        ax1.axhline(y=0, color='gray', linestyle='--')
        ax1.set_xticks(np.arange(self.num_actions + 1))

        ax1.violinplot(dataset=np.transpose(self.reward_distributions), showmeans=True)
        if self.show_true_distributions:
            for patch in ax1.collections:
                patch.set_alpha(0.1)
        else:
            for patch in ax1.collections:
                patch.set_alpha(0.0)

        ax1.violinplot(dataset=self._empirical_rewards, showmeans=True)

        # Plot the rewards over time
        ax2.margins(x=0)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)

        ax2.plot(self._rew_list, color='orange', marker='.')

        plt.show()

    def generate_animation(self):

        # TODO: Streamline!!!
        # TODO: Fix displaying plot below animation

        rc('animation', html='jshtml')
        print("Generating animation...")

        empirical_rewards = np.zeros((self.num_actions, 1)).tolist()
        cum_reward = 0
        rew_list = [0]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), dpi=72)

        def animate(t):
            ax1.clear()
            ax2.clear()

            if t > 0:
                action, reward = self._history[t - 1]

                empirical_rewards[action].append(reward)
                rew_list.append(reward)

                nonlocal cum_reward
                cum_reward += reward

                cum_reward_formatted = "{:.2f}".format(cum_reward)
                fig.suptitle(f'Multi-armed bandit\nCumulative reward: {cum_reward_formatted}', fontsize=14)

                ax1.set_xlabel(f'Action ({int(action) + 1})', fontsize=12)

            else:
                fig.suptitle('Multi-armed bandit\nCumulative reward: 0', fontsize=14)
                ax1.set_xlabel('Action (-)', fontsize=12)

            ax1.set_ylabel('Reward\ndistribution', fontsize=12)

            ax1.violinplot(dataset=np.transpose(self.reward_distributions), showmeans=True)
            if self.show_true_distributions:
                for patch in ax1.collections:
                    patch.set_alpha(0.1)
            else:
                for patch in ax1.collections:
                    patch.set_alpha(0.0)

            ax1.violinplot(dataset=empirical_rewards, showmeans=True)

            ax1.axhline(y=0, color='gray', linestyle='--')
            ax1.set_xticks(np.arange(self.num_actions + 1))

            ax2.margins(x=0)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.plot(rew_list, color='orange', marker='.')

            return fig,  # Note the comma here to return a tuple of artists

        ani = animation.FuncAnimation(fig, animate, frames=np.arange(len(self._history) + 1), blit=True)
        ani.to_jshtml()

        return ani

    def close(self):
        plt.close('all')


gym.envs.register(
    id='MultiArmedBandit-v0',
    entry_point='multi_armed_bandit.env:MultiArmedBandit',
    kwargs={}
)
