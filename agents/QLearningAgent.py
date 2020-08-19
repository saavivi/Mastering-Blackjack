from agents.BaseAgent import BaseAgent
import numpy as np
from collections import defaultdict
from lib.constants import TRAINING_DURATION, NUM_SHOW
from itertools import count
import sys
import random


class QLearningAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        print("Hey from QlearningAgent")

    def train(self):
        def update_Q_sarsamax(alpha, gamma, Q, state, action, reward,
                              next_state=None):
            """Returns updated Q-value for the most recent experience."""
            current = Q[state][action]  # estimate in Q-table (for current state, action pair)
            Qsa_next = np.max(Q[next_state])  # value of next state
            target = reward + (gamma * Qsa_next)  # construct TD target
            new_value = current + (alpha * (target - current))  # get updated value
            return new_value

        def epsilon_greedy(Q, state, nA, eps):
            """Selects epsilon-greedy action for supplied state.

            Params
            ======
                Q (dictionary): action-value function
                state (int): current state
                nA (int): number actions in the environment
                eps (float): epsilon
            """
            if random.random() < eps or Q[state][0] == Q[state][1]:  # select greedy action with probability epsilon
                return random.choice(np.arange(nA))
            else:  # otherwise, select an action randomly
                return np.argmax(Q[state])

        def q_learning(env, num_episodes, alpha=1, gamma=1, epsmin=0):
            """Q-Learning - TD Control

            Params
            ======
                num_episodes (int): number of episodes to run the algorithm
                alpha (float): learning rate
                gamma (float): discount factor
                plot_every (int): number of episodes to use when calculating average score
            """
            nA = env.action_space.n  # number of actions
            Q = defaultdict(
                lambda: np.zeros(nA))  # initialize empty dictionary of arrays

            for i_episode in range(1, num_episodes + 1):
                # monitor progress
                if i_episode % NUM_SHOW == 0:
                    print("\rEpisode {}/{}".format(i_episode, num_episodes),
                          end="")
                    sys.stdout.flush()
                score = 0  # initialize score
                state = env.reset()  # start episode
                eps = max(1.0 / i_episode, epsmin)  # set value of epsilon

                while True:
                    action = epsilon_greedy(Q, state, nA, eps)  # epsilon-greedy action selection
                    next_state, reward, done, _info = env.step(action)  # take action A, observe R, S'
                    score += reward  # add reward to agent's score
                    Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state)
                    state = next_state  # S <- S'
                    # note: no A <- A'
                    if done:
                        break
            policy = dict((k, np.argmax(v)) for k, v in Q.items())
            return Q, policy

        q, self.policy = q_learning(self._env, TRAINING_DURATION)


if __name__ == "__main__":
    b = QLearningAgent()
    b.train()
    b.plot_policy()
    b.play()
