import csv
import numpy as np
import matplotlib.pyplot as plt
import os

ALGO_COLORS = ['blue', 'blue', 'blue', 'red', 'red', 'red', 'magenta', 'magenta', 'magenta']
ALGO_STYLES = ['-', '--', '.-']
ARCH_COLORS = ['brown', 'c', 'g']
EXP_COLORS = ['peru', 'darkseagreen', 'violet']

def load_train_results(f_name):
    data = []
    train_indices = []
    sample_indices = []
    with open(f_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Each row is [i_sample, i_train, eps_results...]
            i_sample = int(row[0]) * 1e-6
            i_train = int(row[1]) * 300
            row = [float(el) / 600 for el in row[2:]]

            data.append(row)
            train_indices.append(i_train)
            sample_indices.append(i_sample)

        # print(len(data[0]))
        rets = np.array(data, dtype=float)

    return rets, train_indices, sample_indices


def setBoxColors(bp, n_boxes, colors):
    for n in range(n_boxes):
        plt.setp(bp['boxes'][n], color=colors[n])
        plt.setp(bp['caps'][n * 2:(n + 1) * 2], color=colors[n])
        plt.setp(bp['whiskers'][n * 2:(n + 1) * 2], color=colors[n])
        plt.setp(bp['fliers'][n * 2:(n + 1) * 2], color=colors[n], marker='x')
        plt.setp(bp['medians'][n], color=colors[n])
    '''
    plt.setp(bp['boxes'][1], color='red')
    plt.setp(bp['caps'][2:4], color='red')
    plt.setp(bp['whiskers'][2:4], color='red')
    plt.setp(bp['fliers'][2:4], color='red')
    plt.setp(bp['medians'][1], color='red')
    '''
    # plt.setp(bp['boxes'][2], color='green')
    # plt.setp(bp['caps'][4:6], color='green')
    # plt.setp(bp['whiskers'][4:6], color='green')
    # plt.setp(bp['fliers'][4:6], color='green')
    # plt.setp(bp['medians'][2], color='green')


def load_test_results(f_name):
    # --- Load test results -----
    #rewards = []  # ntestruns x ntimesteps = 20x500000
    mu_tests = []  # ntestruns x 1
    test_indices = []  # ntestruns x 1
    if os.path.isfile(f_name):
        with open(f_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row[1] == '0':  # if the agent was tested multiple times in the same test environment this column is set to #repretitions.
                    continue  # here we only take into account one repretition per test environment
                # Each row has [random seed of env, test episode with that id, mean reward, reward at every step...]
                i_env = int(row[0])
                mu_test = float(row[2])
                # rews = [round(float(r),4) for r in row[3:]]
                # plt.figure(3)
                # plt.plot(rews[:2000])
                # data.append(rews)

                mu_tests.append(mu_test)  # 20 x 1 (a learned model is tested on 20 distinct test environments, i.e. random seeds)
                test_indices.append(i_env)  # 20 x 1
    else:
        mu_tests = np.NaN
        test_indices = np.NaN
    return mu_tests, test_indices


def load_track_results(f_name):
    # --- Load test results -----
    rewards = []  # ntestruns x ntimesteps = 20x500000
    mu_tests = []  # ntestruns x 1
    test_indices = []  # ntestruns x 1
    if os.path.isfile(f_name):
        with open(f_name, 'r') as f:
            reader = csv.reader(f)
            i_env = 100
            for row in reader:
                # [mean reward, reward at every step...]
                mu_test = float(row[0])

                #rews = [round(float(r),4) for r in row[1:]]
                #rewards.append(rews)
                # plt.figure(3, figsize=(20,20))
                # plt.plot(np.convolve(rews[:10000], np.ones((1000,))/1000, mode='valid'))  #rews[:2000])
                i_env += 1

                mu_tests.append(mu_test)  # 20 x 1 (a learned model is tested on 20 distinct test environments, i.e. random seeds)
                test_indices.append(i_env)  # 20 x 1
    else:
        mu_tests = np.NaN
        test_indices = np.NaN
        rewards = np.NaN
    return mu_tests, test_indices # , np.mean(rewards, axis=0)
