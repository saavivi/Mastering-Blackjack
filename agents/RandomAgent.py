import random
from agents.BaseAgent import BaseAgent
from lib.constants import NUM_HANDS
# from lib.constants import TRAINING_DURATION, EVALUATE_NUM_OF_HANDS,EVALUATE_EVERY
from lib.constants import *
from lib.utils import tournament


class RandomAgent(BaseAgent):

    def __init__(self, alpha=0.015, log_dir='./experiments/random_agent_results/'):
        super().__init__(log_dir=log_dir)
        self.alpha = alpha

    def train(self):
        for i in range(0, TRAINING_DURATION // EVALUATE_EVERY + 1):
            self.logger.log_performance(i * EVALUATE_EVERY, tournament(self._env, EVALUATE_NUM_OF_HANDS)[0])
        self.policy = self.eval_policy


def ra_run_experiments():
    for i in range(NUM_EXP):
        random_agent = RandomAgent(log_dir=f"{RANDOM_RES_DIR}/{i}")
        random_agent.train()
        random_agent.plot(f"RandomAgent_{i}")
    BaseAgent.plot_avg(RANDOM_RES_DIR, "RandomAgent")


if __name__ == "__main__":
    for i in range(NUM_EXP):
        random_agent = RandomAgent(log_dir=f"{RANDOM_RES_DIR}/{i}")
        random_agent.train()
        random_agent.plot(f"RandomAgent_{i}")
        from lib.plotting import plot_avg

        csv_path_list = [f"{RANDOM_RES_DIR}/{j}/performance.csv" for j in
                         range(NUM_EXP)]
        label_names = [f"RA_{j}" for j in range(NUM_EXP)]
        plot_avg(csv_path_list, label_names, "RA_Average",
                 f"{RANDOM_RES_DIR}/avg_fig.png")
