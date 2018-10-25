#!/usr/local/bin/python3.6
#  parse args:
# --method $MTHD --logdir $LOGDIR --seed $SEED --test_env $TEST_ENV --total_timesteps $TOTAL_TESTSTEPS"
#
import os, sys
import argparse, logging
import PPO.eval_ppo_model, A2C.eval_model, DQN.eval_dqn_model
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='The base method of the agent', type=str, default='DQN')
    parser.add_argument('--logdir', help='The directory where final models are located and results are stored', type=str,
                        default='/home/mara/Desktop/logs/DQN_test/dqn_output1')
    # parser.add_argument('--seed', help='Random Seed of the test environment. Nonstationarities are pregenerated, hence seeds 100-119 are possible.', type=float, default=100)
    parser.add_argument('--test_env', help='Type of test environment, describing the respective experiment.', default="ContFlappyBird-v1")
    parser.add_argument('--total_timesteps', help='Number of interactions in the test sequence.', type=int, default=5e2)
    parser.add_argument('--eval_model', help='defines which saved model is evaluated', type=str, default='final')
    parser.add_argument('--result_file', help='name of test result csv file. None if not specifiec', default='lala.csv')
    args = parser.parse_args()
    print(args)

    exp_dir = args.logdir
    # output_dirs = os.listdir(args.logdir)
    #
    # # print(output_dirs)
    # for o_dir in output_dirs:
    #     print('output_dir: '+o_dir)
    #     # if not os.path.isdir(o_dir):
    #     #     os.makedirs(o_dir)

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(exp_dir, 'test.log'))  # create file first
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if 'output' in exp_dir and not 'msub' in exp_dir:
        # args.logdir = os.path.join(exp_dir, exp_dir)
        logger.info('logdir: '+args.logdir)
        if 'PPO' in args.method :
            eval_fcn = PPO.eval_ppo_model.eval_model
        elif 'A2C' in args.method:
            eval_fcn = A2C.eval_model.eval_model
        elif 'DQN' in args.method:
            eval_fcn = DQN.eval_dqn_model.eval_model

        if 'LSTM' in args.method:
            architecture = 'lstm'
        elif 'GRU' in args.method:
            architecture = 'gru'
        else:
            architecture = 'ff'

        for seed in range(100, 120):
            logger.info('Run models in %s with seed %s' % (args.logdir, seed))
            avg_perf, var_perf, max_perf = eval_fcn(render=False,
                                                    nepisodes=1,
                                                    test_steps=args.total_timesteps,
                                                    save_traj=True,
                                                    result_file=args.result_file,
                                                    eval_model=args.eval_model,
                                                    seed=seed,
                                                    test_env=args.test_env,
                                                    architecture=architecture,
                                                    logdir=args.logdir)

# for every output dir:
    # init test_env with random seed
    # take final_model and evaluate it for 500000 steps
        # save reward in every step
    # save average reward

# return


if __name__ == '__main__':
    main()