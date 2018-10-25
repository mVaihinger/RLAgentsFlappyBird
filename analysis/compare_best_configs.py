import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import logging
import sys, os
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
from run_ple_utils import make_ple_env
from A2C import eval_model
from DQN import eval_dqn_model
from DRQN import drqn_eval_model


# TODO check whether step number is different in every agent. It shouln't as there is a fixed update interval..
# TODO set ticks regularly

LOGDIR = '/media/mara/OS/Users/Mara/Documents/Masterthesis/Results'
# LOGDIR = '/home/fr/fr_fr/fr_mv135/logs'

# TODO choose hyperparameter optimization strategy
optimizer='smac'
# optimizer='bohb'

if optimizer == 'bohb':
    DATADIR = os.path.join(LOGDIR, 'config_tests_results', 'BOHB')
    result_dir = os.path.join(LOGDIR, 'config_tests_results', 'BOHB')
else:
    DATADIR = os.path.join(LOGDIR, 'config_tests_results', 'SMAC')
    result_dir = os.path.join(LOGDIR, 'config_tests_results', 'SMAC')

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

logger = logging.getLogger()
logger.propagate = False
fh = logging.FileHandler(os.path.join(result_dir, 'analysis.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)

mthds = os.listdir(DATADIR)
print(mthds)
logger.info(mthds)

plt.ioff()
plot_best = False
fig_algos, ax_algos = plt.subplots(2,2, figsize=(15, 14))
algo_data = []
algo_means = []
algo_stds = []


for mthd in mthds:
# for mthd in ['PPO', 'LSTM_PPO', 'GRU_PPO']:
    if not os.path.isdir(os.path.join(DATADIR,mthd)):
        print(mthd)
        continue
    # if optimizer == 'bohb' and mthd == 'GRU_DQN':
    #     continue

    # if mthd == 'GRU_A2C': # or mthd == 'LSTM_A2C':  # some how results haven'nt' been saved..
    #     continue
    # if mthd == 'LSTM_A2C' or mthd == 'GRU_A2C' or mthd == 'LSTM_DQN' or mthd == 'GRU_DQN' or mthd =='A2C' or mthd == 'DQN':  # some how results haven'nt' been saved..
    #     continue

    MTHDDIR = os.path.join(DATADIR, mthd)

    # TODO output dir - a2c_output / dqn_output

    config_dates = os.listdir(MTHDDIR)
    config_dates.sort()

    config_data = []
    config_means = []
    config_stds = []

    sample_indices = []
    train_indices = []

    fig_configs, ax_configs = plt.subplots(1, 2, figsize=(20, 6))
    fig_configs.suptitle(mthd)

    fig_agents, ax_agents = plt.subplots(len(config_dates), 2, figsize=(20, 12))
    fig_agents.suptitle(mthd)

    if plot_best:
        fig_agents_best, ax_agents_best = plt.subplots(len(config_dates), 1, figsize=(20, 12))
        fig_agents_best.suptitle(mthd)

    eps_name_flag = 1

    for c in range(len(config_dates)):
        if config_dates[c] == 'wrong_runs':
            continue

        # if mthd == 'A2C':
        #     if config_dates[c].find('_09_07') >= 0:
        #         pass
        #     else:
        #         continue
        print(config_dates[c])
        if config_dates[c].find('_08_15_') >= 0:
            print('breaked')
            break
        logger.info('-----------------------------------------')
        logger.info('               Config %s              ' % c)
        logger.info('-----------------------------------------')
        # Load csvs from /logs/algo/date/algo_outputSeed
        date = config_dates[c]
        agent_data = []
        agent_means = []
        agent_stds = []
        logger.info('Config date: %s' % date)

        algo_path = os.path.join(MTHDDIR, date)
        logger.info('Path: %s' % algo_path)

        a = []
        for n in ['a2c_output*', 'dqn_output*', 'ppo_output*']:
            a.extend(glob.glob(os.path.join(algo_path, n)))
        a.sort()

        for agent_result_path in a:
            print(agent_result_path)
            idx = agent_result_path.find('output')
            seed = agent_result_path[idx + 6:]
            logger.info('Agent: %s' % agent_result_path)
            # Load csv
            data = []
            labels = []
            train_indices = []
            sample_indices = []
            with open(os.path.join(agent_result_path, 'train_results.csv'), 'r') as f:
                csvfile = csv.reader(f)
                neps = 0
                for row in csvfile:
                    label = row[0]
                    i_sample = int(row[0]) * 1e-6
                    i_train = int(row[1]) * 300
                    # print('%s %s' % (label, i_sample))

                    row = [float(el)/3000 for el in row[2:]]
                    neps = len(row)
                    labels.append(label)
                    data.append(row)
                    train_indices.append(i_train)
                    sample_indices.append(i_sample)

            # Fishy trick: Add initial values
            data.insert(0, [-2700./3000.] * neps)
            labels.insert(0, 0)
            train_indices.insert(0, 0)
            sample_indices.insert(0, 8e-3)  # TODO adapt this parameter to

            rets = np.array(data, dtype=float)
            mean_data = np.mean(rets, axis=-1)
            std_data = np.std(rets, axis=-1)
            # x = range(len(mean_data))
            if len(train_indices) > 200:
                skip_idx = int(len(train_indices) / 200)  # skip as many samples as needed to get 100 datapoints in the end.
            else:
                skip_idx = 1
            y = mean_data[::skip_idx]
            y1 = mean_data[::skip_idx] - std_data[::skip_idx]
            y2 = mean_data[::skip_idx] + std_data[::skip_idx]
            # print('Agent: %s %s' % (len(x), len(y)))

            if len(config_dates) == 1:
                ax_agents[0].plot(train_indices[::skip_idx], y, label=('seed %s' % seed))  # TODO only set integer ticks
                ax_agents[0].fill_between(train_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
            else:
                ax_agents[c, 0].plot(train_indices[::skip_idx], y, label=('seed %s' % seed))  # TODO only set integer ticks
                ax_agents[c, 0].fill_between(train_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)

            if plot_best:
                # Get performance of the best models of that agent.
                if mthd == 'A2C' or mthd == 'RA2C':
                    eval_fcn = eval_model.eval_model
                elif mthd == 'DQN':
                    eval_fcn = eval_dqn_model.eval_model
                elif mthd == 'DRQN':
                    eval_fcn = drqn_eval_model.eval_model

                model_labels, models_performances, models_performances_std = eval_fcn(render=0, nepisodes=5,
                                                                                      test_steps=3000,
                                                                                      env='FlappyBird-v1',
                                                                                      seed=1,
                                                                                      logdir=agent_result_path,
                                                                                      eval_model='analysis')
                idx = [l.find('_model') for l in model_labels]
                idx_sample = [int(l[idx + 6:]) for l in model_labels]
                indices = np.concatenate((sample_indices[::skip_idx], idx_sample))
                per_mean = np.concatenate((mean_data[::skip_idx], models_performances))
                per_std = np.concatenate((std_data[::skip_idx], models_performances_std))

                # Sort samples
                indices, per_mean, per_std = zip(*sorted(zip(indices, per_mean, per_std)))

                # y = per_mean[::skip_idx]
                # y1 = mean_data[::skip_idx] - std_data[::skip_idx]
                # y2 = mean_data[::skip_idx] + std_data[::skip_idx]
                if len(config_dates) == 1:
                    ax_agents[1].plot(train_indices[::skip_idx], y, label=('seed %s' % seed))  # TODO only set integer ticks
                    ax_agents[1].fill_between(train_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
                else:
                    ax_agents[c, 1].plot(indices, per_mean,
                                         label=('seed %s' % seed))  # TODO only set integer ticks
                    ax_agents[c, 1].fill_between(indices, y1=per_mean - per_std, y2=per_mean + per_std, alpha=0.4)

                ax_agents_best[c, 0].plot(idx_sample, models_performances)
                ax_agents_best[c, 0].fill_between(idx_sample, y1=per_mean - per_std, y2=per_mean + per_std, alpha=0.4)
            else:
                if len(config_dates) == 1:
                    ax_agents[1].plot(train_indices[::skip_idx], y, label=('seed %s' % seed))  # TODO only set integer ticks
                    ax_agents[1].fill_between(train_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
                else:
                    ax_agents[c, 1].plot(sample_indices[::skip_idx], y,
                                         label=('seed %s' % seed))  # TODO only set integer ticks
                    ax_agents[c, 1].fill_between(sample_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)

            # TODO: test whether sorted data is different from unsorted data.
            print('%s: %s' % (mthd, sample_indices))

            agent_data.append(rets)
            agent_means.append(mean_data)  # nagents x 14 = 3 x 14
            agent_stds.append(std_data)

            # Get results of data of best models:
        if len(config_dates) == 1:
            ax_agents[0].legend()
            ax_agents[0].grid()
            ax_agents[0].set_title('Configuration %s, Compare Agents' % str(c))
            ax_agents[0].set_xlabel('Gradient Updates')
            ax_agents[0].set_ylabel('Performance')
            ax_agents[1].legend()
            ax_agents[1].grid()
            ax_agents[1].set_title('Configuration %s, Compare Agents' % str(c))
            ax_agents[1].set_xlabel('Interactions with Env [1e6]')
            # start, end = ax_agents[c, 1].get_xlim()
            # ax_agents[c, 1].set_xticks(np.arange(start, end, 100))
            ax_agents[1].set_ylabel('Performance')
        else:
            ax_agents[c, 0].legend()
            ax_agents[c, 0].grid()
            ax_agents[c, 0].set_title('Configuration %s, Compare Agents' % str(c))
            ax_agents[c, 0].set_xlabel('Gradient Updates')
            ax_agents[c, 0].set_ylabel('Performance')
            ax_agents[c, 1].legend()
            ax_agents[c, 1].grid()
            ax_agents[c, 1].set_title('Configuration %s, Compare Agents' % str(c))
            ax_agents[c, 1].set_xlabel('Interactions with Env [1e6]')
            # start, end = ax_agents[c, 1].get_xlim()
            # ax_agents[c, 1].set_xticks(np.arange(start, end, 100))
            ax_agents[c, 1].set_ylabel('Performance')

        # fig_agents_samples.savefig(os.path.join(result_dir, ('%s_config%s_samples.pdf' % (algo, str(c)))))
        # plt.close(fig_agents_samples)

        # Reduce results to same size
        min_length = min([len(el) for el in agent_data])
        agent_data = [el[:min_length] for el in agent_data]
        agent_means = [el[:min_length] for el in agent_means]
        agent_stds = [el[:min_length] for el in agent_stds]
        indices = sample_indices[:min_length]
        t_indices = train_indices[:min_length]

        mean_config = np.mean(agent_means, axis=0)  # 1 x 14
        std_config = np.std(agent_stds, axis=0)
        # x = range(len(mean_config))
        print(len(mean_config))
        if len(mean_config) > 200:
            skip_idx = int(len(mean_config) / 200)  # skip as many samples as needed to get 100 datapoints per sequence in the end.
        else:
            skip_idx = 1
        # skip_idx = 1

        y = mean_config[::skip_idx]
        y1 = mean_config[::skip_idx] - std_config[::skip_idx]
        y2 = mean_config[::skip_idx] + std_config[::skip_idx]

        print('Confi: %s %s' % (len(mean_config), len(y)))
        print('%s - clipped: %s' % (mthd, indices))

        ax_configs[0].plot(t_indices[::skip_idx], y, label=('Config %s' % c))  # TODO only set integer ticks
        ax_configs[0].fill_between(t_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
        ax_configs[1].plot(indices[::skip_idx], y, label=('Config %s' % c))  # TODO only set integer ticks
        ax_configs[1].fill_between(indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)

        config_data.append(agent_data)
        config_means.append(mean_config)  # Should be 2 x 14
        config_stds.append(std_config)  # Should be 2 x 14

    for ax in ax_configs:
        ax.legend()
        ax.grid()
        ax.set_title('Compare Configs')
        ax.set_ylabel('Performance')
    ax_configs[0].set_xlabel('Gradient Updates')
    ax_configs[1].set_xlabel('Interactions with Env [1e6]')
    print('save fig_agents')

    fig_agents.savefig(os.path.join(result_dir, ('%s_configs.pdf' % mthd)))
    print('saved fig_agents')
    plt.close(fig_agents)
    print('Closed fig_agents')

    # plt.show(fig_configs)
    fig_configs.savefig(os.path.join(result_dir, ('%s.pdf' % mthd)))
    print('saved fig_configs')
    plt.close(fig_configs)
    print('closed fig_config')

    if plot_best:
        fig_agents_best.savefig(os.path.join(result_dir, ('%s_configs_best.pdf' % mthd)))
        print('saved fig_agents_best')
        plt.close(fig_agents_best)
        print('Closed fig_agents_best')

    logger.info('Dim1: %s' % len(config_data))
    logger.info('Dim2: %s' % len(config_data[0]))
    logger.info('Dim3: %s' % len(config_data[0][0]))
    logger.info('Dim4: %s\n' % len(config_data[0][0][0]))

    # Reduce results to same size
    min_length = min([len(el) for el in config_means])
    config_data = [el[:min_length] for el in config_data]
    config_means = [el[:min_length] for el in config_means]
    config_stds = [el[:min_length] for el in config_stds]
    indices = sample_indices[:min_length]
    t_indices = train_indices[:min_length]

    print('%s - clipped: %s' % (mthd, indices))

    mean_algo = np.mean(config_means, axis=0)
    std_algo = np.std(config_stds, axis=0)
    x = range(len(mean_algo))
    print('x: %s' % x[0])
    if len(x) > 200:
        skip_idx = int(len(x) / 200)  # skip as many samples as needed to get 100 datapoints per sequence in the end.
    else:
        skip_idx = 1
    # skip_idx = 1

    y = mean_algo[::skip_idx]
    y1 = mean_algo[::skip_idx] - std_algo[::skip_idx]
    y2 = mean_algo[::skip_idx] + std_algo[::skip_idx]

    print('Algos: %s %s' % (len(x), len(y)))

    ax_algos[0,0].plot(t_indices[::skip_idx], y, label=('%s' % mthd))
    ax_algos[0,0].fill_between(t_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
    ax_algos[0,1].semilogx(indices[::skip_idx], y, label=('%s' % mthd))
    ax_algos[0,1].fill_between(indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)

    # get best config
    best_config = np.argmax(np.mean(config_means, -1))
    y = config_means[best_config][::skip_idx]
    y1 = config_means[best_config][::skip_idx] - config_stds[best_config][::skip_idx]
    y2 = config_means[best_config][::skip_idx] + config_stds[best_config][::skip_idx]

    ax_algos[1, 0].plot(t_indices[::skip_idx], y, label=('%s' % mthd))
    ax_algos[1, 0].fill_between(t_indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)
    ax_algos[1, 1].semilogx(indices[::skip_idx], y, label=('%s' % mthd))
    ax_algos[1, 1].fill_between(indices[::skip_idx], y1=y1, y2=y2, alpha=0.4)

    # ax_algos[0].plot(x, mean_algo, label=('%s' % mthd))
    # ax_algos[0].fill_between(x, y1=mean_algo - std_algo, y2=mean_algo + std_algo, alpha=0.4)
    # ax_algos[1].plot(indices, mean_algo, label=('%s' % mthd))
    # ax_algos[1].fill_between(indices, y1=mean_algo - std_algo, y2=mean_algo + std_algo, alpha=0.4)
    algo_data.append(config_data)
    algo_means.append(mean_algo)
    algo_stds.append(std_algo)
    # ma = []
    # sa = []
    # for a in config_data:
    #     # a = np.swapaxes(np.array(a), axis1=0, axis2=1)
    #     logger.info('Dim1: %s' % len(a))
    #     logger.info('Dim2: %s %s %s' % (len(a[0]), len(a[1]), len(a[2]))) #, len(a[2]),len(a[3])))
    #     logger.info('Dim3: %s %s %s\n' % (len(a[0][0]),len(a[1][0]), len(a[2][0]))) #,len(a[1][0]),len(a[1][0])))
    #     min_length = min([len(el) for el in a])
    #     a = [el[:min_length] for el in a]
    #     logger.info('Dim1: %s' % len(a))
    #     logger.info('Dim2: %s %s %s' % (len(a[0]), len(a[1]), len(a[2])))  # , len(a[2]),len(a[3])))
    #     logger.info('Dim3: %s %s %s\n' % (len(a[0][0]), len(a[1][0]), len(a[2][0])))  # ,len(a[1][0]),len(a[1][0])))
    #     m1 = np.mean(np.array(a), axis=0)
    #     s1 = np.std(np.array(a), axis=0)
    #     logger.info('Dim1: %s' % len(m1))
    #     logger.info('Dim2: %s\n' % len(m1[0]))
    #     m2 = np.mean(m1, axis=-1)
    #     s2 = np.std(s1, axis=-1)
    #     logger.info('Dim1: %s\n' % len(m2))
    #     m3 = np.mean(m2)
    #     s3 = np.std(s2)
    #     ma.append(m3)
    #     sa.append(s3)
    #
    # mean_algo = np.array(ma)
    # std_algo = np.array(sa)  # [np.std(np.std(np.std(np.array(a), axis=-1), axis=-1)) for a in config_results])
    # mean_algo = np.mean(np.mean(np.mean(np.array(config_results), axis=0), axis=0), axis=-1)  # np.array(ma)
    # std_algo = np.std(np.std(np.std(np.array(config_results), axis=0), axis=0), axis=-1)  # np.array(sa)  # [np.std(np.std(np.std(np.array(a), axis=-1), axis=-1)) for a in config_results])

for ax in ax_algos[0]:
    ax.legend()
    ax.grid()
    ax.set_title('Compare algos w.r.t mean performance of all configs')
    ax.set_ylabel('Performance')
for ax in ax_algos[1]:
    ax.grid()
    ax.legend()
    ax.set_title('Compare best configs of algos')
    ax.set_ylabel('Performance')
ax_algos[0,0].set_xlabel('Gradient Updates')
ax_algos[0,1].set_xlabel('Interactions with Env [1e6]')
# ax_algos[0,1].set_xlim([1e-20,1e2])
ax_algos[1,0].set_xlabel('Gradient Updates')
ax_algos[1,1].set_xlabel('Interactions with Env [1e6]')
plt.show(fig_algos)
fig_algos.savefig(os.path.join(result_dir, 'algos.pdf'))
plt.close(fig_algos)