import random
import numpy as np
from collections import namedtuple


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Memory(object):
    """
    Memory for saving transitions
    """

    def __init__(self, memory_size, batch_size):
        """
        Initialize
        Args:
            memory_size (int): the size of the memory buffer
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        """
        Save transition into memory
        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        """
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        """
        Sample a mini-batch from the replay memory
        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        """
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))
