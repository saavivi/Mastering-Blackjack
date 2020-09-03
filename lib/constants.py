CARDS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
SUITS = ('H', 'D', 'C', 'S')

TRAINING_DURATION = int(5e5)
NUM_SHOW = int(1e4)
NUM_HANDS = int(4e5)

DQN_TRAINING_DURATION = int(1e3)
EVALUATE_EVERY = TRAINING_DURATION // 100
EVALUATE_NUM_OF_HANDS = ((len(CARDS) * len(SUITS)) ** 2)*10

