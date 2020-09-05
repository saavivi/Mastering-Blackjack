import rlcard
from DRL.DQNAgent import DQNAgent
from lib.Logger import Logger
from lib.utils import tournament
from lib.constants import EVALUATE_EVERY, DQN_TRAINING_DURATION, EVALUATE_NUM_OF_HANDS
from lib.constants import *

if __name__ == "__main__":
    for i in range(NUM_EXP):
        # Make environment
        env = rlcard.make('blackjack', config={'seed': i})
        eval_env = rlcard.make('blackjack', config={'seed': i})

        # Set the iterations numbers and how frequently we evaluate/save plot


        # The intial memory size
        memory_init_size = 100

        # Train the agent every X steps
        train_every = 1

        # The paths for saving the logs and learning curves
        log_dir = f"{DQN_RES_DIR}/{i}"

          # Set up the agents
        agent = DQNAgent('dqn',
                         action_num=env.action_num,
                         replay_memory_init_size=memory_init_size,
                         train_every=train_every,
                         state_shape=env.state_shape,
                         mlp_layers=[128, 256, 512],
                         debug=True)
        env.set_agents([agent])
        eval_env.set_agents([agent])


        # Init a Logger to plot the learning curve
        logger = Logger(log_dir, debug=True)

        for episode in range(DQN_TRAINING_DURATION):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % EVALUATE_EVERY == 0:
                logger.log_performance(env.timestep, tournament(eval_env, EVALUATE_NUM_OF_HANDS)[0])

            # Close files in the logger
            # logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')

