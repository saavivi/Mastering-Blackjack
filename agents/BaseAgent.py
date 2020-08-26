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
        wins, draws, losses = 0, 0, 0
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
                    elif reward == 0:
                        draws += 1
                    else:
                        losses += 1
                    break
        print("")
        print(f"Wins {wins}/{i_episode}.")
        print(f"Draws {draws}/{i_episode}.")
        print(f"Losses {losses}/{i_episode}.")
        avg_reward = (wins-losses) / num_plays
        print(f"Average Reward: {avg_reward}")

    def plot_policy(self):
        assert self.policy is not None
        plot_policy(self.policy)

    def plot_value_function(self):
        assert self.policy is not None
        plot_value_function(self.q)






