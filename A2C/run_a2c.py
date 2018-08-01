#!/usr/local/bin/python3.6

import sys, os
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))

from run_ple_utils import make_ple_envs, make_ple_env, arg_parser, params_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy, LargerLSTMPolicy
from A2C.eval_model import eval_model
from A2C.A2C_OAI_NENVS import learn

import logging
import datetime, os

# Run this function in SMAC script. It takes the arguments from the function call and sets unset
# arguments to their default value.


def a2c_params_parser(**kwargs):
    param_dict = params_parser()
    param_dict.add_cat_param("policy", options=['mlp', 'casta', 'largemlp', 'lstm'], default='largemlp', dtype=str)
    param_dict.add_num_param("nenvs", lb=1, ub=30, default=3, dtype=int)
    param_dict.add_num_param("nsteps", lb=1, ub=500, default=50, dtype=int)
    param_dict.add_num_param("vf_coeff", lb=1e-2, ub=1., default=0.2, dtype=float)
    param_dict.add_num_param("ent_coeff", lb=1e-12, ub=1., default=1e-7, dtype=float)
    param_dict.add_num_param("gamma", lb=0.01, ub=1, default=0.90, dtype=float)
    param_dict.add_num_param("lr", lb=1e-12, ub=1, default=5e-4, dtype=float)
    param_dict.add_cat_param("lrschedule", options=['constant', 'linear', 'double_linear_con'], default='constant',
                            dtype=str)
    param_dict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    param_dict.add_num_param("units_shared_layer1", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_shared_layer2", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_policy_layer", lb=1, ub=700, default=64, dtype=int)
    return param_dict.check_params(**kwargs)


def run_a2c_smac(**kwargs):
    params = a2c_params_parser(**kwargs)
    # paramDict = ParamDict()
    # paramDict.add_cat_param("env", options=['FlappyBird-v1'], default='FlappyBird-v1', dtype=str)
    # paramDict.add_num_param("total_timesteps", lb=0, ub=10e15, default=int(10e3), dtype=int)
    # paramDict.add_num_param("seed", lb=0, ub=np.inf, default=123, dtype=int),
    # paramDict.add_cat_param("policy", options=['mlp', 'casta', 'largemlp'], default='largemlp', dtype=str)
    # paramDict.add_num_param("nenvs", lb=1, ub=16, default=3, dtype=int),
    # paramDict.add_num_param("nsteps", lb=1, ub=100, default=50, dtype=int),
    # paramDict.add_num_param("vf_coeff", lb=1e-2, ub=0.4, default=0.2, dtype=float),
    # paramDict.add_num_param("ent_coeff", lb=1e-9, ub=1e-2, default=1e-7, dtype=float),
    # paramDict.add_num_param("gamma", lb=0.5, ub=0.99, default=0.90, dtype=float),
    # paramDict.add_num_param("lr", lb=1e-9, ub=1e-2, default=5e-4, dtype=float),
    # paramDict.add_cat_param("lrschedule", options=['constant', 'linear', 'double_linear_con'], default='constant', dtype=str),
    # paramDict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    # paramDict.add_num_param("units_shared_layer1", lb=8, ub=260, default=64, dtype=int),
    # paramDict.add_num_param("units_shared_layer2", lb=8, ub=260, default=64, dtype=int),
    # paramDict.add_num_param("units_policy_layer", lb=8, ub=260, default=64, dtype=int),
    # paramDict.add_num_param("log_interval", lb=1, ub=1e5, default=100, dtype=int),
    # # paramDict.add_num_param("save_interval", lb=1, ub=1e5, default=1000, dtype=int),
    # paramDict.add_num_param("show_interval", lb=0, ub=1e5, default=0, dtype=int),
    # paramDict.add_cat_param("logdir", options=None, default='/home/mara/Desktop/logs/A2C_OAI_NENVS', dtype=str),
    # paramDict.add_cat_param("eval_model", options=['all', 'final'], default='all', dtype=str)
    # params = paramDict.check_params(**kwargs)

    # logger = logging.getLogger(__name__)
    # logger.propagate = False  # no duplicate logging outputs
    # fh = logging.FileHandler(os.path.join(params["logdir"], 'run.log'))
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    # logger.addHandler(fh)

    seed = params["seed"]
    # print(params["env"], params["nenvs"])  # TODO remove this here!
    ple_env = make_ple_envs(params["env"], num_env=params["nenvs"], seed=seed)
    test_env = make_ple_env(params["test_env"], seed=seed)

    # print(ple_env)

    if params["policy"] == 'mlp':
        policy_fn = MlpPolicy
    elif params["policy"] == 'casta':
        policy_fn = CastaPolicy
    elif params["policy"] == 'largemlp':
        policy_fn = LargerMLPPolicy
    elif params["policy"] == 'lstm':
        policy_fn =LargerLSTMPolicy
    else:
        print('Policy option %s is not implemented yet.' % params["policy"])

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        for k, v in params.items():
            f.write(k + ': ' + str(v) + '\n')
    logging.info(datetime.time)
    learn(policy_fn,
          env=ple_env,
          test_env=test_env,
          seed=seed,
          nsteps=params["nsteps"],
          vf_coef=params["vf_coeff"],
          ent_coef=params["ent_coeff"],
          gamma=params["gamma"],
          lr=params["lr"],
          lrschedule=params["lrschedule"],
          max_grad_norm=params["max_grad_norm"],
          units_per_hlayer=(params["units_shared_layer1"],
                            params["units_shared_layer2"],
                            params["units_policy_layer"]),
          log_interval=params["log_interval"],
          test_interval=params["test_interval"],
          total_timesteps=params["total_timesteps"],
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
    # parser = arg_parser()
    # parser.add_argument('--nenvs', help='Number of environments', type=int, default=3)
    # parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'casta', 'largemlp'], default='largemlp')

    # parser.add_argument('--log_interval', help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ', type=int, default=30)
    # # parser.add_argument('--save_interval', help='Model is saved after <save_interval> model updates', type=int, default=1000)
    # parser.add_argument('--show_interval', help='Env is rendered every n-th episode. 0 = no rendering', type=int, default=0)
    # parser.add_argument('--logdir', help='directory where logs are stored', default='/home/mara/Desktop/logs/A2C_OAI_NENVS')  # '/mnt/logs/A2C')
    # args = parser.parse_args()
    parser = arg_parser()
    parser.add_argument('--nenvs', help='Number of environments', type=int, default=3)
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'casta', 'largemlp', 'lstm'],
                        default='largemlp')
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float,
                        default=0.01)
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--nsteps', help='n environment steps per train update', type=int, default=50)
    parser.add_argument('--vf_coeff', help='Weight of value function loss in total loss', type=float, default=0.2)
    parser.add_argument('--ent_coeff', help='Weight of entropy in total loss', type=float, default=1e-7)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule', choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--units_shared_layer1', help='Units in first hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_shared_layer2', help='Units in second hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_policy_layer', help='Units in hidden layer in policy head', type=int, default=64)
    args = parser.parse_args()

    # args.total_timesteps = 50000
    # args.show_interval = 2
    # args.env = 'FlappyBird-v3' # TODO!!
    seed = args.seed
    # print(args.env, args.nenvs)
    env = make_ple_envs(args.env, num_env=args.nenvs, seed=seed)
    test_env = make_ple_env(args.test_env, seed=seed)
    # print(env)

    if args.policy == 'mlp':
        policy_fn = MlpPolicy
    elif args.policy == 'casta':
        policy_fn = CastaPolicy
    elif args.policy == 'largemlp':
        policy_fn = LargerMLPPolicy
    elif args.policy == 'lstm':
        policy_fn =LargerLSTMPolicy
    else:
        print('Policy option %s is not implemented yet.' % args.policy)

    # store hyperparms setting
    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)

    a2c_output_dir = os.path.join(args.logdir, ('a2c_output'+str(args.seed)))
    if not os.path.isdir(a2c_output_dir):  # TODO check what this does
        os.makedirs(a2c_output_dir)

    with open(os.path.join(a2c_output_dir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()  # TODO setup root logger is necessary to use FileHandler
    fh = logging.FileHandler(os.path.join(a2c_output_dir, 'algo.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    learn(policy_fn,
          env=env,
          test_env=test_env,
          seed=seed,
          nsteps=args.nsteps,
          vf_coef=args.vf_coeff,
          ent_coef=args.ent_coeff,
          gamma=args.gamma,
          lr=args.lr,
          lrschedule=args.lrschedule,
          max_grad_norm=args.max_grad_norm,
          units_per_hlayer=(args.units_shared_layer1,
                            args.units_shared_layer2,
                            args.units_policy_layer),
          log_interval=args.log_interval,
          test_interval=args.test_interval,
          total_timesteps=args.total_timesteps,
          logdir=a2c_output_dir,
          keep_model=args.keep_model) # TODO remove
    env.close()

    args.logdir = a2c_output_dir
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=20, **args.__dict__)

    with open(os.path.join(args.logdir, 'hyperparams.txt'), 'a') as f:
        f.write('\n')
        f.write('Results: \n')
        f.write('average performance: ' + str(avg_perf) + '\n')
        f.write('performance variance: ' + str(var_perf) + '\n')
        f.write('maximum return: ' + str(max_return) + '\n')
    # return avg_perf, var_perf, max_return


if __name__ == '__main__':
    main()
