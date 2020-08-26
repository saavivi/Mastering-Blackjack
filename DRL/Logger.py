import os
import csv
import matplotlib.pyplot as plt


class Logger(object):
    """
    Logger saves the running results and helps make plots from the results
    """

    def __init__(self, log_dir, debug=False):
        """ Initialize the labels, legend and paths of the plot and log file.
        Args:
            log_path (str): The path the log files
        """
        self.debug = debug
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # self.txt_file = open(self.txt_path, 'w')
        # self.csv_file = open(self.csv_path, 'w')
        self.fieldnames = ['timestep', 'reward']
        with open(self.csv_path, 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
        with open(self.txt_path, 'w+') as txt_file:
            txt_file.flush()

    def log(self, text):
        """ Write the text to log file then print it.
        Args:
            text(string): text to log
        """
        with open(self.txt_path, 'a+') as txt_file:
            txt_file.write(text+'\n')
            txt_file.flush()
        if self.debug:
            print(text)

    def log_performance(self, timestep, reward):
        """ Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        """
        with open(self.csv_path, 'a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow({'timestep': timestep, 'reward': reward})
        if self.debug:
            print('')
        self.log('----------------------------------------')
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')

    def plot(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm)

    # def close_files(self):
    #     """ Close the created file objects
    #     """
    #     if self.txt_path is not None:
    #         self.txt_file.close()
    #     if self.csv_path is not None:
    #         self.csv_file.close()


def plot(csv_path, save_path, algorithm):
    """ Read data from csv file and plot the results
    """
    with open(csv_path) as csvfile:
        # print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['timestep']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='timestep', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
