from agents.BaseAgent import BaseAgent
import numpy as np
from collections import defaultdict
from lib.constants import *
from lib.utils import tournament
from itertools import count
import sys
import random


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
    if random.random() < eps or Q[state][0] == Q[state][1]:
        return random.choice(np.arange(nA))
    else:
        return np.argmax(Q[state])


def q_learning(env, to_train, already_trained=0, q=None, alpha=1.0, gamma=1.0,
               eps_start=1.0, eps_decay=.99999, eps_min=0.05, using_alpha=True):
    """Q-Learning - TD Control

    Params
    ======
        num_episodes (int): number of episodes to run the algorithm
        alpha (float): learning rate
        gamma (float): discount factor
        plot_every (int): number of episodes to use when calculating average score
    """
    nA = env.action_space.n  # number of actions
    if q is None:
        q = defaultdict(lambda: np.zeros(nA))

    for i_episode in range(1, to_train + 1):
        # monitor progress
        # if i_episode % NUM_SHOW == 0:
        #     print("\rEpisode {}/{}".format(i_episode, num_episodes),
        #           end="")
        #     sys.stdout.flush()
        score = 0  # initialize score
        state = env.reset()  # start episode
        eps = max(1.0 / (already_trained+i_episode), 0.0)  # set value of epsilon

        while True:
            action = epsilon_greedy(q, state, nA,
                                    eps)  # epsilon-greedy action selection
            next_state, reward, done, _info = env.step(
                action)  # take action A, observe R, S'
            score += reward  # add reward to agent's score
            if not using_alpha:
                alpha = eps
            q[state][action] = update_Q_sarsamax(alpha, gamma, q, state,
                                                 action, reward, next_state)
            state = next_state  # S <- S'
            # note: no A <- A'
            if done:
                break
    policy = dict((k, np.argmax(v)) for k, v in q.items())
    return policy, q


class QLearningAgent(BaseAgent):

    def __init__(self, alpha=0.015, log_dir='./experiments/q_learning_results/'):
        super().__init__(log_dir=log_dir)
        self.alpha = alpha

    def train(self):
        for i in range(0, TRAINING_DURATION // EVALUATE_EVERY + 1):
            self.logger.log_performance(i * EVALUATE_EVERY,
                                        tournament(self._env,
                                                   EVALUATE_NUM_OF_HANDS)[0])
            # Best Alpha so far is 0.015
            self.eval_policy, self.q = q_learning(self._env,
                                                  q=self.q,
                                                  to_train=EVALUATE_EVERY,
                                                  already_trained=EVALUATE_EVERY*i,
                                                  alpha=self.alpha,
                                                  gamma=1.0,
                                                  eps_start=1.0,
                                                  eps_decay=0.99999,
                                                  eps_min=0.015,
                                                  using_alpha=True
                                                  )
        self.policy = self.eval_policy


def ql_run_experiments():
    for i in range(NUM_EXP):
        q_agent = QLearningAgent(alpha=0.15, log_dir=f"{Q_LEARNING_RES_DIR}/{i}")
        q_agent.train()
        q_agent.plot_policy(save=True, save_path=f"{Q_LEARNING_RES_DIR}/{i}/policy.png")
        q_agent.plot(f"Q_Learning_{i}")
    BaseAgent.plot_avg(Q_LEARNING_RES_DIR, "QLearning")

if __name__ == "__main__":
    for i in range(NUM_EXP):
        q_agent = QLearningAgent(alpha=0.15, log_dir=f"{Q_LEARNING_RES_DIR}/_{i}")
        q_agent.train()
        q_agent.plot_policy(save=True, save_path=f"{Q_LEARNING_RES_DIR}/{i}/policy.png")
        q_agent.plot(f"Q_Learning_{i}")
        q_agent
    from lib.plotting import plot_avg
    csv_path_list = [f"{Q_LEARNING_RES_DIR}/{j}/performance.csv" for j in
                     range(NUM_EXP)]
    label_names = [f"QL_{j}" for j in range(NUM_EXP)]
    plot_avg(csv_path_list, label_names, "QL_Average", f"{Q_LEARNING_RES_DIR}/avg_fig.png")


