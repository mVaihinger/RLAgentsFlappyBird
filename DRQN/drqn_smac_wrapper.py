import logging
import numpy as np
import sys, os, glob

# Import SMAC utilities
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

sys.path.append(os.path.dirname(sys.path[0]))
from DRQN.run_drqn import run_smac
from run_ple_utils import arg_parser


def dqn_smac_wrapper(**params):

    logdir = params["logdir"]

    dqn_output_dir = os.path.join(logdir, 'dqn_rnn_output' + str(params["instance_id"]))
    if not os.path.isdir(dqn_output_dir):
        os.makedirs(dqn_output_dir)
    smac_output_dir = os.path.join(logdir, 'smac3_output' + str(params["instance_id"]))

    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)
    #args.logdir = logdir

    var = 0
    max_ret = 0

    def dqn_from_cfg(cfg):
        """ Creates the A2C algorithm based on the given configuration.

        :param cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        :return: A quality score of the algorithms perfromance
        """

        # For deactivated paraeters the configuration stores None-values
        # This is not accepted by the a2c algorithm, hence we remove them.
        cfg = {k: cfg[k] for k in cfg if cfg[k]}

        # create run directory
        dir_list = glob.glob(os.path.join(dqn_output_dir, 'run*'))
        rundir = 'run{:02d}'.format(len(dir_list)+1)  # 'run' + str(len(dir_list) + 1)

        params["logdir"] = os.path.join(dqn_output_dir, rundir)
        os.makedirs(params["logdir"])
        # print(args.logdir)
        avg_perf, var_perf, max_return = run_smac(**params, **cfg)
        logger.info('average performance: %s' % avg_perf)
        logger.info('performance variance: %s' % var_perf)
        logger.info('maximum episode return: %s' % max_return)

        var = var_perf
        max_ret = max_return

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
    # TODO which parameters to optimize?
    # gamma = UniformFloatHyperparameter("gamma", 0.7, 1, default_value=0.95)
    epsilon = UniformFloatHyperparameter("epsilon", 0.1, 1, default_value=0.9)                # initial epsilon
    epsilon_decay = UniformFloatHyperparameter("epsilon_decay", 0.2, 0.999, default_value=0.795)  # decay rate
    # lr = UniformFloatHyperparameter("lr", 2e-5, 8e-3, default_value=5e-4)
    # nbatch = UniformIntegerHyperparameter("nbatch", 2, 50, default_value=4)   # number of episodes from which mb traces are drawn.
    # trace_length = UniformIntegerHyperparameter("trace_length", 1, 30, default_value=8)  # length of each mb_trace
    units_shared_layer1 = UniformIntegerHyperparameter("units_layer1", 8, 400, default_value=64)
    units_shared_layer2 = UniformIntegerHyperparameter("units_layer2", 1, 400, default_value=150)
    units_policy_layer = UniformIntegerHyperparameter("units_layer3", 1, 400, default_value=120)
    cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
                            epsilon, epsilon_decay]) # , gamma, lr, nbatch, trace_length])

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
        if params["instance_id"] == 1:
            f.write("---------------------------------------------\n")
            f.write("               %s                " % params["logdir"][28:])
            f.write("---------------------------------------------\n")
        f.write("Optimized hyperparameters of smac instance %s\n" % params["instance_id"])
        for k in optimized_cfg:
            f.write(str(k) + ': ' + str(optimized_cfg[k]) + '\n')
        f.write("Optimized performance - average episode return: %.2f\n" % optimized_performance)
        f.write("Variance of average episode return %s \n" % var)
        f.write("Maximal episode return: %s \n\n" % max_ret)

    return optimized_cfg


def dqn_arg_parser():
    parser = arg_parser()
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--tau', help='Update rate of target network', type=float, default=0.99)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule',
                        choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--nbatch',
                        help='Batch size. Number of sampless drawn from buffer, which are used to update the model.',
                        type=int, default=3)
    parser.add_argument('--buffer_size', help='Replay buffer size', type=int, default=5000)
    parser.add_argument('--trace_length', help='Length of the traces obtained from the batched episodes', type=int,
                        default=8)
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float,
                        default=0.01)
    parser.add_argument('--update_interval', type=int, default=5,
                        help='Frequency with which the network model is updated based on minibatch data.')
    return parser.parse_args()


def main():
    args = dqn_arg_parser()
    # print("rhs: " + str(args.run_parallel))
    _ = dqn_smac_wrapper(**args.__dict__)


if __name__ == '__main__':
    main()

