import numpy as np
import torch


def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector
    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.
    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs


def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment
    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.
    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.player_num)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs

def set_global_seed(seed):
    ''' Set the global see for reproducing results
    Args:
        seed (int): The seed
    Note: If using other modules with randomness, they also need to be seeded
    '''
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
