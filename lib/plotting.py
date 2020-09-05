import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_value_function(Q, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        fig1.close(fig1)
    else:
        fig1.show()
        # plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig2.close()
    else:
        fig2.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        fig3.close()
    else:
        fig3.show()

    return fig1, fig2, fig3


def plot_policy(policy, save=False, save_path=None):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in policy:
            return policy[x, y, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array(
            [[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1,
                         extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    if not save:
        plt.show()
    else:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)


def plot_avg(csv_path_list, labels_list, title, save_path):
    """
    Read data from csv file and plot the results to save_path
    :param csv_path: path to csv file with the data
    :param save_path: path where to save the figure
    :param algorithm: name of the algorithm used
    """
    plt.figure()
    avg_x = None
    avg_y = None
    i = 0
    for csv_path, label in zip(csv_path_list, labels_list):
        with open(csv_path) as csvfile:
            i += 1
            # print(csv_path)
            reader = csv.DictReader(csvfile)
            xs = []
            ys = []
            for row in reader:
                xs.append(int(row['timestep']))
                ys.append(float(row['reward']))

            if avg_x is None:
                avg_x, avg_y = xs, ys
            else:
                avg_x = [ele1 + ele2 for ele1, ele2 in zip(avg_x, xs)]
                avg_y = [ele1 + ele2 for ele1, ele2 in zip(avg_y, ys)]

            plt.plot(xs, ys, label=label)

    avg_x = [lst_ele/i for lst_ele in avg_x]
    avg_y = [lst_ele/i for lst_ele in avg_y]
    plt.plot(avg_x, avg_y, label="Average", color='black', marker='.')

    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.close()




from lib.constants import *
if __name__ == "__main__":
    csv_path_list = [f"{MC_RES_DIR}/{i}/performance.csv" for i in range(NUM_EXP)]
    label_names = [f"MC_{i}" for i in range(5)]
    print(csv_path_list)
    print(label_names)
    plot_avg(csv_path_list, label_names, "aa", './experiments/mc_results/avg_fig.png')
