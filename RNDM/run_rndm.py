import os,sys
# import logging
# from collections import deque
import numpy as np
import random
import csv
import argparse
import logging

print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
from run_ple_utils import arg_parser, make_ple_env


# This file implements the "training" and evaluation of a random agent

# def eval_rndm(env, seed, total_timesteps, show_interval, logdir):
#     env.reset()
#     total_return = 0
#     reward_window = deque(maxlen=100)
#
#     result_path = os.path.join(logdir, 'test_results.csv')
#
#     for t in range(total_timesteps):
#         if show_interval > 0:
#             env.render()
#         obs, reward, dones, _ = env.step(random.choice([0,1]))
#         reward_window.append(reward)
#         total_return += reward
#
#         # save to csv
#         with open(result_path, "a") as csvfile:
#             writer = csv.writer(csvfile)
#             row = [reward, np.mean(reward_window)]
#             writer.writerow(row)
#
#     return total_return / total_timesteps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_env', help='testv environment ID', default='ContFlappyBird-v3')
    parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(2e2))
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--logdir', default='/home/mara/Desktop/logs/RND',
                        help='directory where logs are stored')
    parser.add_argument('--show_interval', type=int, default=0,
                        help='Env is rendered every n-th episode. 0 = no rendering')
    parser.add_argument('--eval_model', choices=['all', 'inter', 'final'], default='inter',
                        help='Eval all stored models, only the final model or only the intermediately stored models (while testing the best algorithm configs)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Init test_results.csv
    for i, p_flap in zip(range(1, 4), [0.1, 0.3, 0.5]):

        rnd_output_dir = os.path.join(args.logdir, ('rnd_output' + str(i)))
        if not os.path.isdir(rnd_output_dir):
            os.makedirs(rnd_output_dir)

        logger = logging.getLogger()
        fh = logging.FileHandler(os.path.join(rnd_output_dir, 'algo.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        result_path = os.path.join(rnd_output_dir, 'test_results.csv')

        for s in range(100, 120):
            logger.info('make env with seed %s' % s)
            test_env = make_ple_env(args.test_env, seed=s)

            test_env.reset()
            # logger.info('reset')
            total_return = 0
            rew_traj =[]

            t = 0
            while t < args.total_timesteps:
                t+=1
                if args.show_interval > 0:
                    test_env.render()
                    # logger.info('render')
                # logger.info('step')
                obs, reward, dones, _ = test_env.step(np.random.choice([0, 1], p=[p_flap, 1-p_flap]))
                # logger.info('stepped')
                # reward_window.append(reward)
                total_return += reward
                rew_traj.append(reward)
            test_env.close()

            with open(result_path, "a") as csvfile:
                logger.info('write csv')
                writer = csv.writer(csvfile)
                rew_traj[0:0] = [s, 0, np.mean(rew_traj)]
                writer.writerow(rew_traj)


if __name__ == '__main__':
    main()