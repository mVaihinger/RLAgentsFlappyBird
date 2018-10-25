import logging
import numpy as np
import csv
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, LessThanCondition

# Import SMAC utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import sys, os, datetime, glob
sys.path.append(os.path.dirname(sys.path[0]))
from DQN.run_dqn import run_dqn_smac
from run_ple_utils import smac_parser


# def arg_parser():
#     import argparse
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # TODO add buffer size
#     # parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
#     # parser.add_argument('--epsilon', help='Epsilon for epsilon-greedy policy', type=float, default=0.5)
#     # parser.add_argument('--epsilon_decay', help='Epsilon decay rate', type=float, default=0.995)
#     # parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
#     parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule',
#                         choices=['constant', 'linear', 'double_linear_con'], default='constant')
#     parser.add_argument('--buffer_size', help='Replay buffer size', type=float, default=5000)
#     parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float,
#                         default=0.01)
#     parser.add_argument('--log_interval',
#                         help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ',
#                         type=int, default=500)
#     parser.add_argument('--save_interval', help='Model is saved after <save_interval> model updates', type=int,
#                         default=500)
#     parser.add_argument('--show_interval', help='Env is rendered every n-th episode. 0 = no rendering', type=int,
#                         default=0)
#     parser.add_argument('--logdir', help='directory where logs are stored',
#                         default='/home/mara/Desktop/logs/DQN')  # '/mnt/logs/A2C')
#     parser.add_argument('--seed', help='RNG seed', type=int, default=0)
#     parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(2e5))
#     parser.add_argument('--runcount_limit', help='amount of algorithm evaluations allowed to optimize hyperparameters',
#                         type=int, default=int(3))
#     parser.add_argument('--eval_model', help='eval all stored models or only final model', choices=['all', 'final'],
#                         default='all')
#     parser.add_argument('--run_parallel', help='flag which determine whethe smac instances are run in parallel or not.',
#                         choices=["True", "true", "False", "false"], type=str, default="false")
#     parser.add_argument('--instance_id', help='id of the smac instance', type=int, default=1)
#     args = parser.parse_args()
#     return args

def dqn_arg_parser():
    parser = smac_parser()
    # parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--batch_size', help='Batch size. Number of sampless drawn from buffer, which are used to update the model.',
                        type=int, default=50)
    # parser.add_argument('--tau', help='Update rate of target netowrk', type=float, default=0.99)
    # parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--buffer_size', help='Replay buffer size', type=int, default=int(5000))
    # parser.add_argument('--trace_length', help='Length of the traces obtained from the batched episodes', type=int,
    #                     default=1)
    parser.add_argument('--update_interval', type=int, default=5, help='Network parameters are updated after N real intercation with the environemnt')
    #                     help='Frequency with which the network model is updated based on minibatch data.')
    return parser.parse_args()


def dqn_smac_wrapper(**params):

    logdir = params["logdir"]

    dqn_output_dir = os.path.join(logdir, 'dqn_output{:02d}'.format(params["instance_id"]))
    if not os.path.isdir(dqn_output_dir):
        os.makedirs(dqn_output_dir)
    smac_output_dir = os.path.join(logdir, 'smac3_output{:02d}'.format(params["instance_id"]))

    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)
    #args.logdir = logdir

    def dqn_from_cfg(cfg):
        """ Creates the A2C algorithm based on the given configuration.

        :param cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        :return: A quality score of the algorithms perfromance
        """

        # For deactivated parameters the configuration stores None-values
        # This is not accepted by the a2c algorithm, hence we remove them.
        cfg = {k: cfg[k] for k in cfg if cfg[k]}

        # create run directory
        dir_list = glob.glob(os.path.join(dqn_output_dir, 'run*'))
        rundir = 'run{:02d}'.format(len(dir_list)+1)  # 'run' + str(len(dir_list) + 1)

        params["logdir"] = os.path.join(dqn_output_dir, rundir)
        os.makedirs(params["logdir"])
        # print(args.logdir)
        avg_perf, var_perf, max_return = run_dqn_smac(**params, **cfg)
        logger.info('average performance: %s' % avg_perf)
        logger.info('performance variance: %s' % var_perf)
        logger.info('maximum episode return: %s' % max_return)

        score = - avg_perf  # - (avg_perf - 0.2 * var_perf + 0.5 * max_return)  # SMAC is minimizing this.
        logger.info('Quality measure of the current learned agent: %s\n' % score)
        return score

    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    logger = logging.getLogger()
    logger.propagate = False  # no duplicate logging outputs
    fh = logging.FileHandler(os.path.join(logdir, 'smac.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)

    # Build configuration space and define all hyperparameters
    cs = ConfigurationSpace()
    epsilon = UniformFloatHyperparameter("epsilon", 0.2, 0.9, default_value=0.6)                # initial epsilon
    epsilon_decay = UniformFloatHyperparameter("epsilon_decay", 0.2, 1, default_value=0.995)  # decay rate
    lr = UniformFloatHyperparameter("lr", 0.0005, 0.01, default_value=0.005)
    units_shared_layer1 = UniformIntegerHyperparameter("units_layer1", 8, 100, default_value=24)
    units_shared_layer2 = UniformIntegerHyperparameter("units_layer2", 8, 100, default_value=24)
    units_policy_layer = UniformIntegerHyperparameter("units_layer3", 8, 100, default_value=24)
    activ_fcn = CategoricalHyperparameter("activ_fcn", ['relu6', 'elu', 'mixed'], default_value='relu6')
    gamma = UniformFloatHyperparameter("gamma", 0.6, 0.90, default_value=0.80)
    tau = UniformFloatHyperparameter("tau", 0.5, 1., default_value=0.7)
    # update_interval = UniformIntegerHyperparameter("update_interval", 1, 300, default_value=50)
    if params["architecture"] == 'lstm' or (params["architecture"] == 'gru'):
        trace_length = UniformIntegerHyperparameter("trace_length", 1, 20, default_value=8)
        # buffer_condition = LessThanCondition(child=trace_length, parent=params["buffer_size"])
        # pa["batch_size"] = 5
        cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
                                epsilon, epsilon_decay, activ_fcn, lr, gamma, tau, trace_length])
    else:
        params.pop("batch_size")
        batch_size = UniformIntegerHyperparameter("batch_size", 1, 100, default_value=30)
        # buffer_condition = LessThanCondition(child=batch_size, parent=params["buffer_size"], value=33)
        # InCondition(child=batch_size, value=33)
        cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
                                epsilon, epsilon_decay, activ_fcn, lr, gamma, tau, batch_size])

    # Create scenario object
    logger.info('##############################################')
    logger.info('Setup SMAC instance')
    logger.info('##############################################')

    logger.info('Output_dir: %s' % smac_output_dir)
    if params["run_parallel"].lower() == "true":
        scenario = Scenario({"run_obj": "quality",      # we optimize quality of learned agent
                             "runcount-limit": params["runcount_limit"],     # Maximum function evaluations
                             "cs": cs,                  # configutation space
                             "deterministic": "true",
                             "output_dir": smac_output_dir,
                             "shared_model": True,
                             "input_psmac_dirs": os.path.join(logdir, 'smac3_output*')
                             })
    else:
        scenario = Scenario({"run_obj": "quality",  # we optimize quality of learned agent
                             "runcount-limit": params["runcount_limit"],  # Maximum function evaluations
                             "cs": cs,  # configutation space
                             "deterministic": "true",
                             "output_dir": smac_output_dir,
                             })
    seed = np.random.RandomState(params["seed"])
    smac = SMAC(scenario=scenario, rng=seed, tae_runner=dqn_from_cfg)

    logger.info('##############################################')
    logger.info('Run Optimization')
    logger.info('##############################################')

    optimized_cfg = smac.optimize()

    logger.info('##############################################')
    logger.info('Evaluate Configuration found by SMAC')
    logger.info('##############################################')

    optimized_performance = dqn_from_cfg(optimized_cfg)
    logger.info("Optimized config")
    for k in optimized_cfg:
        logger.info(str(k) + ": " + str(optimized_cfg[k]))
    logger.info("Optimized performance: %.2f" % optimized_performance)

    with open(os.path.join(logdir, 'opt_hyperparams.txt'), 'a') as f:
        for k in optimized_cfg:
            f.write(str(k) + ': ' + str(optimized_cfg[k]) + '\n')
        f.write("Optimized performance: %.2f\n\n" % optimized_performance)

    with open(os.path.join(logdir, 'opt_hyperparams.csv'), 'a') as f:
        labels = []
        for k in optimized_cfg:
            labels.append(str(k))
        labels.insert(0, 'performance')
        labels.insert(0, 'instance_id')
        writer = csv.DictWriter(f, fieldnames=labels)
        if params["instance_id"] == 1:
            writer.writeheader()
        optimized_cfg._values["performance"] = optimized_performance
        optimized_cfg._values["instance_id"] = params["instance_id"]
        writer.writerow(optimized_cfg._values)

    return optimized_cfg

def main():
    args = dqn_arg_parser()
    # print("rhs: " + str(args.run_parallel))
    args.architecture = 'lstm'

    _ = dqn_smac_wrapper(**args.__dict__)

if __name__ == '__main__':
    main()

    # args = arg_parser()
    #
    # logdir = os.path.join(args.logdir, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # os.makedirs(logdir)
    # args.logdir = logdir
    # optimized_cfg = dqn_smac_wrapper(**args.__dict__)
