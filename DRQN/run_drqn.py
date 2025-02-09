#!/usr/local/bin/python3.6
import logging
import sys, os
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
from run_ple_utils import make_ple_env, arg_parser, params_parser
from DRQN.drqn import q_learning
from DRQN.drqn_eval_model import eval_model


def dqn_params_parser(**kwargs):
    param_dict = params_parser()
    param_dict.add_num_param("gamma", lb=0.01, ub=1., default=0.95, dtype=float)
    param_dict.add_num_param("epsilon", lb=0.01, ub=1., default=0.99, dtype=float)
    param_dict.add_num_param("epsilon_decay", lb=0.01, ub=1., default=0.995, dtype=float)
    param_dict.add_num_param("tau", lb=0.1, ub=1., default=0.99, dtype=float)
    param_dict.add_num_param("lr", lb=1e-12, ub=1., default=5e-4, dtype=float)
    param_dict.add_cat_param("lrschedule", options=['constant', 'linear', 'double_linear_con'], default='constant', dtype=str)
    param_dict.add_num_param("nbatch", lb=1, ub=100, default=4, dtype=int)
    param_dict.add_num_param("trace_length", lb=1, ub=100, default=8, dtype=int)
    param_dict.add_num_param("buffer_size", lb=1, ub=1e6, default=int(4000), dtype=int)
    param_dict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    param_dict.add_num_param("units_layer1", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_layer2", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_layer3", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("update_interval", lb=1, ub=1000, default=5, dtype=int)
    return param_dict.check_params(**kwargs)


# Run this function in SMAC script. It takes the arguments from the function call and sets unset
# arguments to their default value.
def run_smac(**kwargs):
    params = dqn_params_parser(**kwargs)
    seed = params["seed"]
    ple_env = make_ple_env(params["env"], seed=seed)
    test_env = make_ple_env(params["test_env"], seed=seed)

    # with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
    #     f.write('KWARGS\n')
    #     for k, v in kwargs.items():
    #         f.write(k + ': ' + str(v) + '\n')

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        f.write('PARAMS\n')
        for k, v in params.items():
            f.write(k + ': ' + str(v) + '\n')

    # print(params)

    q_learning(ple_env,
               test_env=test_env,
               seed=seed,
               total_timesteps=params["total_timesteps"],
               gamma=params["gamma"],
               epsilon=params["epsilon"],
               epsilon_decay=params["epsilon_decay"],
               tau=params["tau"],
               lr=params["lr"],
               lrschedule=params["lrschedule"],
               buffer_size=params["buffer_size"],
               nbatch=params["nbatch"],
               trace_length=params["trace_length"],
               max_grad_norm=params["max_grad_norm"],
               units_per_hlayer=(params["units_layer1"],
                                 params["units_layer2"],
                                 params["units_layer3"]),
               update_interval=params["update_interval"],
               log_interval=params["log_interval"],
               test_interval=params["test_interval"],
               show_interval=params["show_interval"],
               logdir=params["logdir"],
               keep_model=params["keep_model"])

    ple_env.close()
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=10, **params)

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        f.write('\n')
        f.write('Results: \n')
        f.write('average performance: ' + str(avg_perf) + '\n')
        f.write('performance variance: ' + str(var_perf) + '\n')
        f.write('maximum return: ' + str(max_return) + '\n')

    return avg_perf, var_perf, max_return


def main():
    parser = arg_parser()
    # parser = arg_parser()
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--epsilon', help='Epsilon for epsilon-greedy policy', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', help='Epsilon decay rate', type=float, default=0.995)
    parser.add_argument('--tau', help='Update rate of target netowrk', type=float, default=0.99)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule', choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--nbatch', help='Batch size. Number of sampless drawn from buffer, which are used to update the model.',
                        type=int, default=3)
    parser.add_argument('--buffer_size', help='Replay buffer size', type=int, default=10)
    parser.add_argument('--trace_length', help='Length of the traces obtained from the batched episodes', type=int, default=8)
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float, default=0.01)
    parser.add_argument('--units_layer1', help='Units in first hidden layer', type=int, default=64)
    parser.add_argument('--units_layer2', help='Units in second hidden layer', type=int, default=64)
    parser.add_argument('--units_layer3', help='Units in third hidden layer', type=int, default=64)
    parser.add_argument('--update_interval', type=int, default=5,
                        help='Frequency with which the network model is updated based on minibatch data.')
    # parser.add_argument('--log_interval', help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ', type=int, default=30)
    # parser.add_argument('--show_interval', help='Env is rendered every n-th episode. 0 = no rendering', type=int, default=30)
    # parser.add_argument('--logdir', help='directory where logs are stored', default='/home/mara/Desktop/logs/A2C_OAI_NENVS')  # '/mnt/logs/A2C')
    args = parser.parse_args()

    seed = args.seed
    env = make_ple_env(args.env, seed=seed)
    test_env = make_ple_env(args.env, seed=seed)

    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)

    dqn_output_dir = os.path.join(args.logdir, ('dqn_rnn_output' + str(args.seed)))
    if not os.path.isdir(dqn_output_dir):  # TODO check what this does
        os.makedirs(dqn_output_dir)

    # store hyperparms setting
    with open(os.path.join(dqn_output_dir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()  # TODO setup root logger is necessary to use FIleHandler
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(dqn_output_dir, 'algo.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


    q_learning(env,
               test_env=test_env,
               seed=seed,
               total_timesteps=args.total_timesteps,
               gamma=args.gamma,
               epsilon=args.epsilon,
               epsilon_decay=args.epsilon_decay,
               tau=args.tau,
               lr=args.lr,
               lrschedule=args.lrschedule,
               buffer_size=args.buffer_size,
               nbatch=args.nbatch,
               trace_length=args.trace_length,
               max_grad_norm=args.max_grad_norm,
               units_per_hlayer=(args.units_layer1,
                                 args.units_layer2,
                                 args.units_layer3),
               update_interval=args.update_interval,
               log_interval=args.log_interval,
               test_interval=args.test_interval,
               show_interval=args.show_interval,
               logdir=dqn_output_dir,
               keep_model=args.keep_model)
    env.close()

    args.logdir = dqn_output_dir
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=15, **args.__dict__)

    with open(os.path.join(args.logdir, 'hyperparams.txt'), 'a') as f:
        f.write('\n')
        f.write('Results: \n')
        f.write('average performance: ' + str(avg_perf) + '\n')
        f.write('performance variance: ' + str(var_perf) + '\n')
        f.write('maximum return: ' + str(max_return) + '\n')
    # return avg_perf, var_perf, max_return


if __name__ == '__main__':
    main()
