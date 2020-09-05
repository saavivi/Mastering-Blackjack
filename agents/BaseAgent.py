import gym
import numpy as np
from collections import defaultdict
from lib.Logger import Logger
from gym.envs.toy_text.blackjack import BlackjackEnv
from lib.constants import *
from lib.plotting import plot_policy, plot_value_function
from lib.plotting import plot_avg
from lib.utils import tournament


class BaseAgent:

    def __init__(self, env=gym.make('Blackjack-v0'), log_dir=None):
        self._env = env
        self.q = defaultdict(lambda: np.zeros(self._env.action_space.n))
        self.policy = None
        self.eval_policy = None
        self.log_dir = log_dir
        self.logger = Logger(self.log_dir, debug=False)
        # Adding run function to Gym env
        if isinstance(self._env, BlackjackEnv):
            def run(is_training=False):
                observation = self._env.reset()
                while True:
                    if (self.eval_policy is None) or \
                            (observation not in self.eval_policy):
                        action = np.random.choice(
                            np.arange(env.action_space.n))
                    else:
                        action = np.argmax(self.eval_policy[observation])
                    observation, reward, done, _ = self._env.step(action)
                    if done:
                        return _, np.asarray([int(reward)])

            self._env.run = run
            self._env.player_num = 1

    def train(self):
        pass

    def play(self, num_plays=NUM_HANDS):
        return tournament(self._env, num_plays)

    def plot_policy(self, save=False, save_path=None):
        assert self.policy is not None
        plot_policy(self.policy, save=save, save_path=save_path)

    def plot_value_function(self):
        assert self.policy is not None
        plot_value_function(self.q)

    def plot(self, algo_name):
        self.logger.plot(algo_name)

    @staticmethod
    def plot_avg(base_dir, algo_name):
        csv_path_list = [f"{base_dir}/{j}/performance.csv" for j in
                         range(NUM_EXP)]
        label_names = [f"{algo_name}_{j}" for j in range(NUM_EXP)]
        plot_avg(csv_path_list, label_names, f"{algo_name}_Average",
                 f"{base_dir}/avg_fig.png")
