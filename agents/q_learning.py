import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting
import matplotlib.pyplot as plt

NUM_EPISODES = int(1e5)
NUM_PLAYS = 100


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    e_greedy_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes),
                  end="")
            sys.stdout.flush()

        # Implement this!
        state = env.reset()
        for t in itertools.count():
            action = np.random.choice([action for action in range(env.nA)],
                                      p=e_greedy_policy(state))
            next_state, reward, done, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state])
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            state = next_state

    return Q, make_epsilon_greedy_policy(Q, 0 , env.nA)


if __name__ == "__main__":

    # Blackjack env
    matplotlib.style.use('ggplot')
    env = BlackjackEnv()
    Q, policy = q_learning(env, num_episodes=NUM_EPISODES)
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        (player_sum, _, _ ) = state
        if player_sum > 21:
            continue
        action_value = np.max(actions)
        V[state] = action_value

    plotting.plot_value_function(V, title="Optimal Value Function")

    wins = 0
    for i_episode in range(1, NUM_PLAYS):
        observation = env.reset()
        for t in range(100):
            # print_observation(observation)
            action = np.argmax(policy(observation))
            # print("Taking action: {}".format( ["Stick", "Hit"][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                if reward == 1.0:
                    wins +=1
                sys.stdout.flush()
                print(f"\rWins {wins}/{i_episode}.")
                sys.stdout.flush()
                # print_observation(observation)
                # print("Game end. Reward: {}\n".format(float(reward)))

                break
