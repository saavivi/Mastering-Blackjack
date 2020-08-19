import gym
import numpy as np
from collections import defaultdict
from itertools import count
import sys
from lib.plotting import plot_policy
from lib.constants import TRAINING_DURATION, NUM_HANDS, NUM_SHOW
from lib.plotting import plot_policy, plot_value_function


class BaseAgent:

    def __init__(self):
        self._env = gym.make('Blackjack-v0')
        self.q = defaultdict(lambda: np.zeros(self._env.action_space.n))
        self.policy = None

    def train(self):
        pass

    def play(self, num_plays=NUM_HANDS):
        wins = 0
        for i_episode in range(1, num_plays):
            observation = self._env.reset()
            while True:
                # print_observation(observation)
                action = np.argmax(self.policy[observation])
                # print("Taking action: {}".format( ["Stick", "Hit"][action]))
                observation, reward, done, _ = self._env.step(action)
                if done:
                    if reward == 1.0:
                        wins += 1
                    sys.stdout.flush()
                    if i_episode % (num_plays - 1) == 0:
                        print(f"\rWins {wins}/{i_episode}.")
                    sys.stdout.flush()
                    # print_observation(observation)
                    # print("Game end. Reward: {}\n".format(float(reward)))

                    break

    def plot_policy(self):
        assert self.policy is not None
        plot_policy(self.policy)

    def plot_value_function(self):
        assert self.policy is not None
        plot_value_function(self.q)






