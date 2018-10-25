import os
import logging
import numpy as np
import tensorflow as tf

from run_ple_utils import arg_parser, make_ple_env, make_ple_envs
from models_OAI import LargerLSTMPolicy, LargerMLPPolicy, GRUPolicy
from utils_OAI import set_global_seeds
from META.maml import meta_learn

import PPO.ppo, A2C.A2C_OAI_NENVS


def main():
    parser = arg_parser()
    parser.add_argument('--early_stop', help='stop bad performing runs ealier', type=bool, default=False)
    parser.add_argument('--nenvs', help='Number of parallel simulation environmenrs', type=int, default=1)
    parser.add_argument('--activ_fcn', choices=['relu6', 'elu', 'mixed'], type=str, default='mixed',
                        help='Activation functions of network layers', )
    parser.add_argument('--lr', help='Learning Rate', type=float, default=0.001)
    parser.add_argument('--nsteps', type=int, default=32, help='number of samples based on which gradient is updated')
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--vf_coeff', help='Weight of value function loss in total loss', type=float, default=0.2)
    parser.add_argument('--ent_coeff', help='Weight of entropy in total loss', type=float, default=7e-5)
    parser.add_argument('--units_shared_layer1', help='Units in first hidden layer which is shared', type=int,
                        default=28)
    parser.add_argument('--units_shared_layer2', help='Units in second hidden layer which is shared', type=int,
                        default=59)
    parser.add_argument('--units_policy_layer', help='Units in hidden layer in policy head', type=int, default=21)

    # PPO args
    parser.add_argument('--nminibatches', help='Number of minibatches per sampled data batch.', type=int, default=2)
    parser.add_argument('--noptepochs',
                        help='Number of optimization epochs with sample data, i.e. how often samples are reused.',
                        type=int, default=4)

    parser.add_argument('--lam', help='Lambda parameter for GAE', type=float, default=0.95)
    parser.add_argument('--cliprange', help='Defines the maximum policy change allowed, before clipping.', type=float,
                        default=0.2)

    # MAML args
    parser.add_argument('--K', help='length of each rollout (=trajectory)', type=int, default=20) # Test how well it works with other measures.
    parser.add_argument('--train_batchsz', help='number of rollouts per adaptation/training update (=fast update)', type=int, default=1)
    parser.add_argument('--kshot', help='number of adaptation/training update (=fast updates) per task between two meta updates', type=int, default=1000)
    parser.add_argument('--test_batchsz', help='number of rollouts with updated model on which test_loss is computed',
                        type=int, default=1)
    parser.add_argument('--meta_batchsz', help='number of sampled tasks per meta update', type=int, default=4)  # parallely or sequentially
    parser.add_argument('--test_stage', help='whether or not meta learner is in test_stage', type=bool, default=False)

    parser.add_argument('--base_agent', help='type of base learning agent, i.e. A2C or PPO agent', type=str, default='ppo')
    args = parser.parse_args()
    print(args)

    ple_env = make_ple_envs(args.env, args.nenvs, seed=args.seed-1)
    ple_test_env = make_ple_env(args.test_env, seed=100 + (args.seed-1))

    if args.architecture == 'ff':
        policy_fn = LargerMLPPolicy
    elif args.architecture == 'lstm':
        policy_fn = LargerLSTMPolicy
    elif args.architecture == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % args.policy)

    output_dir = os.path.join(args.logdir, ('a2c_output'+str(args.seed)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(output_dir, 'algo.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # if not args.test_stage:  # construct training model
    #     pass
    args.env = ple_env
    args.test_env = ple_test_env
    args.logdir = output_dir
    args.units_per_hlayer=(args.units_shared_layer1,
                           args.units_shared_layer2,
                           args.units_policy_layer)
    args.policy = policy_fn

    args.total_timesteps = 200000

    meta_learn(**args.__dict__)
    ple_env.close()


if __name__ == "__main__":
    main()
