#!/usr/local/bin/python3.6
import simplejson
import sys, os
import logging

print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
from run_ple_utils import make_ple_envs, make_ple_env, arg_parser, params_parser
from models_OAI import LargerMLPPolicy, LargerLSTMPolicy, GRUPolicy
from A2C.eval_model import eval_model
from A2C.A2C_OAI_NENVS import learn


# Run this function in SMAC script. It takes the arguments from the function call and sets unset
# arguments to their default value.


def a2c_params_parser(**kwargs):
    param_dict = params_parser()
    # param_dict.add_cat_param("architecture", options=['ff', 'lstm', 'gru'], default='ff', dtype=str)

    # param_dict.add_num_param("lr", lb=1e-12, ub=1, default=5e-4, dtype=float)
    # param_dict.add_num_param("batch_size", lb=1, ub=500, default=50, dtype=int)
    # param_dict.add_num_param("gamma", lb=0.01, ub=1, default=0.90, dtype=float)
    # param_dict.add_cat_param("activ_fcn", options=['relu6', 'elu', 'mixed'], default='relu6', dtype=str)

    param_dict.add_num_param("nenvs", lb=1, ub=30, default=3, dtype=int)
    param_dict.add_num_param("vf_coeff", lb=1e-2, ub=1., default=0.2, dtype=float)
    param_dict.add_num_param("ent_coeff", lb=1e-12, ub=1., default=1e-7, dtype=float)
    # param_dict.add_cat_param("lrschedule", options=['constant', 'linear', 'double_linear_con'], default='constant',
    #                         dtype=str)
    # param_dict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    param_dict.add_num_param("units_shared_layer1", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_shared_layer2", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_policy_layer", lb=1, ub=700, default=64, dtype=int)
    return param_dict.check_params(**kwargs)


def run_a2c_smac(**kwargs):
    params = a2c_params_parser(**kwargs)

    # logger = logging.getLogger(__name__)
    # logger.propagate = False  # no duplicate logging outputs
    # fh = logging.FileHandler(os.path.join(params["logdir"], 'run.log'))
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    # logger.addHandler(fh)

    seed = params["seed"]
    ple_env = make_ple_envs(params["env"], num_env=params["nenvs"], seed=seed)
    test_env = make_ple_env(params["test_env"], seed=3000)

    if params["architecture"] == 'ff':
        policy_fn = LargerMLPPolicy
    elif params["architecture"] == 'lstm':
        policy_fn = LargerLSTMPolicy
    elif params["architecture"] == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % params["policy"])

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        for k, v in params.items():
            f.write(k + ': ' + str(v) + '\n')

    early_stopped = learn(policy_fn,
                          env=ple_env,
                          test_env=test_env,
                          seed=seed,
                          total_timesteps=params["total_timesteps"],
                          log_interval=params["log_interval"],
                          test_interval=params["test_interval"],
                          show_interval=params["show_interval"],
                          logdir=params["logdir"],
                          lr=params["lr"],
                          # lrschedule=params["lrschedule"],
                          max_grad_norm=params["max_grad_norm"],
                          units_per_hlayer=(params["units_shared_layer1"],
                                            params["units_shared_layer2"],
                                            params["units_policy_layer"]),
                          activ_fcn=params["activ_fcn"],
                          gamma=params["gamma"],
                          vf_coef=params["vf_coeff"],
                          ent_coef=params["ent_coeff"],
                          batch_size=params["batch_size"],
                          early_stop=params["early_stop"],
                          keep_model=params["keep_model"])
    ple_env.close()

    if not early_stopped:
        avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=10, test_steps=3000, **params)

        with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
            f.write('\n')
            f.write('Results: \n')
            f.write('average performance: ' + str(avg_perf) + '\n')
            f.write('performance variance: ' + str(var_perf) + '\n')
            f.write('maximum return: ' + str(max_return) + '\n')

        return avg_perf, var_perf, max_return
    else:
        return -3000, 3000, -3000


def main():
    parser = arg_parser()
    parser.add_argument('--early_stop', help='stop bad performing runs ealier', type=bool, default=False)
    parser.add_argument('--nenvs', help='Number of parallel simulation environmenrs', type=int, default=1)
    parser.add_argument('--activ_fcn', choices=['relu6', 'elu', 'mixed'], type=str, default='relu6',
                        help='Activation functions of network layers', )
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=50,
                        help='number of samples based on which gradient is updated', )
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--vf_coeff', help='Weight of value function loss in total loss', type=float, default=0.2)
    parser.add_argument('--ent_coeff', help='Weight of entropy in total loss', type=float, default=1e-7)
    parser.add_argument('--units_shared_layer1', help='Units in first hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_shared_layer2', help='Units in second hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_policy_layer', help='Units in hidden layer in policy head', type=int, default=64)
    args = parser.parse_args()

    seed = args.seed
    # env = make_ple_envs('ContFlappyBird-gfNS-nrf0-train-v0', num_env=args.nenvs, seed=seed - 1)
    env = make_ple_envs(args.env, num_env=args.nenvs, seed=seed - 1)
    test_env = make_ple_env(args.test_env, seed=100 + (seed - 1))

    # TODO:
    # env = make_ple_envs(args.env, num_env=args.nenvs, seed=seed)
    # test_env = make_ple_env(args.test_env, seed=seed)  # is automatically set to seed+100

    if args.architecture == 'ff':
        policy_fn = LargerMLPPolicy
    elif args.architecture == 'lstm':
        policy_fn = LargerLSTMPolicy
    elif args.architecture == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % args.policy)

    # store hyperparms setting
    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)

    a2c_output_dir = os.path.join(args.logdir, ('a2c_output'+str(args.seed)))
    if not os.path.isdir(a2c_output_dir):
        os.makedirs(a2c_output_dir)

    with open(os.path.join(a2c_output_dir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(a2c_output_dir, 'algo.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    early_stopped = learn(policy_fn,
                          env=env,
                          test_env=test_env,
                          seed=seed,
                          total_timesteps=args.total_timesteps,
                          log_interval=args.log_interval,
                          test_interval=args.test_interval,
                          show_interval=args.show_interval,
                          logdir=a2c_output_dir,
                          lr=args.lr,
                          # lrschedule=args.lrschedule,
                          max_grad_norm=args.max_grad_norm,
                          units_per_hlayer=(args.units_shared_layer1,
                                            args.units_shared_layer2,
                                            args.units_policy_layer),
                          activ_fcn=args.activ_fcn,
                          gamma=args.gamma,
                          vf_coef=args.vf_coeff,
                          ent_coef=args.ent_coeff,
                          batch_size=args.batch_size,
                          early_stop=args.early_stop,
                          keep_model=args.keep_model)
    env.close()

    args.logdir = a2c_output_dir
    # avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=5, **args.__dict__)
    #
    # with open(os.path.join(args.logdir, 'hyperparams.txt'), 'a') as f:
    #     f.write('\n')
    #     f.write('Results: \n')
    #     f.write('average performance: ' + str(avg_perf) + '\n')
    #     f.write('performance variance: ' + str(var_perf) + '\n')
    #     f.write('maximum return: ' + str(max_return) + '\n')
    # # return avg_perf, var_perf, max_return


if __name__ == '__main__':
    main()
