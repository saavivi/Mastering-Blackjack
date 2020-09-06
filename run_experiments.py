from agents.MonteCarloAgent import mc_run_experiments
from agents.QLearningAgent import ql_run_experiments
from agents.RandomAgent import ra_run_experiments
from DRL.run_dqn import dqn_run_experiments


if __name__ == "__main__":
    ra_run_experiments()
    mc_run_experiments()
    ql_run_experiments()
    dqn_run_experiments()
