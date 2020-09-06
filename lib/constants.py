CARDS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
SUITS = ('H', 'D', 'C', 'S')

TRAINING_DURATION = int(4e3)
NUM_HANDS = int(4e5)

DQN_TRAINING_DURATION = int(4e3)
EVALUATE_EVERY = TRAINING_DURATION // 100
EVALUATE_NUM_OF_HANDS = ((len(CARDS) * len(SUITS)) ** 2)

Q_LEARNING_RES_DIR = "./experiments/q_learning_results"
MC_RES_DIR = "./experiments/mc_results"
RANDOM_RES_DIR = "./experiments/random_results"
DQN_RES_DIR = "./experiments/dqn_results"

NUM_EXP = 4




