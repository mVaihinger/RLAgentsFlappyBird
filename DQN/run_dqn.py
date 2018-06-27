#!/usr/local/bin/python3.6

# import sys
# sys.path.append('/mnt')
from run_ple_utils import make_ple_env, arg_parser
from eval_dqn_model import eval_model
from DQN_PLE import q_learning

import numpy as np
import logging
import datetime, os

def main():
    parser = arg_parser()
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--epsilon', help='Epsilon for epsilon-greedy policy', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', help='Epsilon decay rate', type=float, default=0.995)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule', choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--batch_size', help='Batch size. Number of sampless drawn from buffer, which are used to update the model.',
                        type=int, default=50)
    parser.add_argument('--buffer_size', help='Replay buffer size', type=float, default=4000)
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float, default=0.01)
    parser.add_argument('--units_layer1', help='Units in first hidden layer', type=int, default=64)
    parser.add_argument('--units_layer2', help='Units in second hidden layer', type=int, default=64)
    parser.add_argument('--units_layer3', help='Units in third hidden layer', type=int, default=64)
    parser.add_argument('--log_interval', help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ', type=int, default=30)
    parser.add_argument('--show_interval', help='Env is rendered every n-th episode. 0 = no rendering', type=int, default=30)
    parser.add_argument('--logdir', help='directory where logs are stored', default='/home/mara/Desktop/logs/A2C_OAI_NENVS')  # '/mnt/logs/A2C')
    args = parser.parse_args()

    seed = args.seed
    env = make_ple_env(args.env, seed=seed)

    logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    os.makedirs(logdir)
    # store hyperparms setting
    with open(os.path.join(logdir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()  # TODO setup root logger is necessary to use FIleHandler
    fh = logging.FileHandler(os.path.join(logdir, 'smac.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    q_learning(env,
               seed=seed,
               total_timesteps=args.total_timesteps,
               gamma=args.gamma,
               epsilon=args.epsilon,
               epsilon_decay=args.epsilon_decay,
               lr=args.lr,
               lrschedule=args.lrschedule,
               max_replay_buffer_size=args.buffer_size,
               max_grad_norm=args.max_grad_norm,
               units_per_hlayer=(args.units_layer1,
                                 args.units_layer2,
                                 args.units_layer3),
               log_interval=args.log_interval,
               show_interval=args.show_interval,
               logdir=logdir)
    env.close()
    # return avg_perf, var_perf, max_return

class ParamDict():
    def __init__(self):
        self.dict = {}

    def add_num_param(self, name, lb, ub, default, dtype):
        self.dict[name] = (lb, ub, default, dtype)

    def add_cat_param(self, name, options, default, dtype):
        self.dict[name] = (options, default, dtype)

    def setDefaults(self, paramsDict):
        for k,v in self.dict.items():
            paramsDict[k] = v[-2]

    def check_type(self, val, dtype):
        return isinstance(val, dtype)

    def check_limits(self, val, *args):
        if len(args) == 1:
            if args[0] is not None:
                return val in args[0]
            else:
                return True  # if no option is given, i.e. in logdir case
        else:
            return (args[0] <= val) & (val < args[1])

    def check_params(self, **kwargs):
        params = {}
        self.setDefaults(params)
        for k,v in kwargs.items():
            if self.dict.get(k):
                if self.check_type(v, self.dict[k][-1]):
                    if self.check_limits(v, *self.dict[k][:-2]):
                        params[k] = v
                    else:
                        print('Argument %s is out of bounds. Value is %s. Should be in %s' % (k, v, self.dict[k][:-2]))
                else:
                    print('Argument %s doesn\'t have expected data type %s' % (k, self.dict[k][-1]))
        return params


# Run this function in SMAC script. It takes the arguments from the function call and sets unset
# arguments to their default value.
def run_dqn_smac(**kwargs):
    paramDict = ParamDict()
    paramDict.add_cat_param("env", options=['FlappyBird-v1'], default='FlappyBird-v1', dtype=str)
    paramDict.add_num_param("total_timesteps", lb=0, ub=10e15, default=int(10e3), dtype=int)
    paramDict.add_num_param("seed", lb=0, ub=np.inf, default=123, dtype=int),
    paramDict.add_num_param("gamma", lb=0.5, ub=0.99, default=0.90, dtype=float),
    paramDict.add_num_param("epsilon", lb=0.01, ub=1, default=0.50, dtype=float),
    paramDict.add_num_param("epsilon_decay", lb=0.2, ub=1, default=0.995, dtype=float)
    paramDict.add_num_param("lr", lb=1e-9, ub=1e-2, default=5e-4, dtype=float),
    paramDict.add_cat_param("lrschedule", options=['constant', 'linear', 'double_linear_con'], default='constant', dtype=str),
    paramDict.add_num_param("batch_size", lb=5, ub=2000, default=128, dtype=int)
    paramDict.add_num_param("buffer_size", lb=1, ub=1e5, default=4000, dtype=int),
    paramDict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    paramDict.add_num_param("units_layer1", lb=8, ub=260, default=64, dtype=int),
    paramDict.add_num_param("units_layer2", lb=1, ub=260, default=64, dtype=int),
    paramDict.add_num_param("units_layer3", lb=1, ub=260, default=64, dtype=int),
    paramDict.add_num_param("log_interval", lb=1, ub=1e5, default=100, dtype=int),
    paramDict.add_num_param("show_interval", lb=0, ub=1e5, default=0, dtype=int),
    paramDict.add_cat_param("logdir", options=None, default='/home/mara/Desktop/logs/A2C_OAI_NENVS', dtype=str),
    paramDict.add_cat_param("eval_model", options=['all', 'final'], default='all', dtype=str)
    params = paramDict.check_params(**kwargs)

    seed = params["seed"]
    ple_env = make_ple_env(params["env"], seed=seed)

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        for k, v in params.items():
            f.write(k + ': ' + str(v) + '\n')

    q_learning(ple_env,
               seed=seed,
               total_timesteps=params["total_timesteps"],
               gamma=params["gamma"],
               epsilon=params["epsilon"],
               epsilon_decay=params["epsilon_decay"],
               lr=params["lr"],
               lrschedule=params["lrschedule"],
               max_replay_buffer_size=params["buffer_size"],
               batch_size=params["batch_size"],
               max_grad_norm=params["max_grad_norm"],
               units_per_hlayer=(params["units_layer1"],
                                 params["units_layer2"],
                                 params["units_layer3"]),
               log_interval=params["log_interval"],
               show_interval=params["show_interval"],
               logdir=params["logdir"])

    ple_env.close()
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=200, **params)

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        f.write('\n')
        f.write('Results: \n')
        f.write('average performance: ' + str(avg_perf) + '\n')
        f.write('performance variance: ' + str(var_perf) + '\n')
        f.write('maximum return: ' + str(max_return) + '\n')

    return avg_perf, var_perf, max_return


if __name__ == '__main__':
    main()
