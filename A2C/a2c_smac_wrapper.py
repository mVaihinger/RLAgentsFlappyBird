import logging
import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
# from ConfigSpace.conditions import InCondition

# Import SMAC utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import sys, os, datetime, glob, time
# sys.path.append('/media/mara/OS/Users/Mara/Documents/Masterthesis/RLAgents/RLAgentsFlappyBird')
# sys.path.append('~/RLAgents_FlappyBird')
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))

from A2C.run_a2c import run_a2c_smac
from run_ple_utils import arg_parser

# TODO check whether normal run works

# check whether parallel run works

def a2c_arg_parser():
    parser = arg_parser()
    parser.add_argument('--nenvs', help='Number of environments', type=int, default=3)
    parser.add_argument('--nsteps', help='n environment steps per train update', type=int, default=50)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule', choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'casta', 'largemlp', 'lstm'],
                        default='lstm')
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float,
                        default=0.01)
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.95)
    return parser.parse_args()


def a2c_smac_wrapper(**params):
    # logdir = os.path.join(params["logdir"], str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # os.makedirs(logdir)
    logdir = params["logdir"]

    a2c_output_dir = os.path.join(logdir, 'a2c_output' + str(params["instance_id"]))
    if not os.path.isdir(a2c_output_dir):
        os.makedirs(a2c_output_dir)
    smac_output_dir = os.path.join(logdir, 'smac3_output' + str(params["instance_id"]))

    def a2c_from_cfg(cfg):
        """ Creates the A2C algorithm based on the given configuration.

        :param cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        :return: A quality score of the algorithms performance
        """
        # For deactivated paraeters the configuration stores None-values
        # This is not accepted by the a2c algorithm, hence we remove them.
        cfg = {k: cfg[k] for k in cfg if cfg[k]}

        # create run directory
        dir_list = glob.glob(os.path.join(a2c_output_dir, 'run*'))
        rundir = 'run{:02d}'.format(len(dir_list)+1) # + str(len(dir_list) + 1)

        params["logdir"] = os.path.join(a2c_output_dir, rundir)
        os.makedirs(params["logdir"])
        avg_perf, var_perf, max_return = run_a2c_smac(**params, **cfg)
        logger.info('average performance: %s' % avg_perf)
        logger.info('performance variance: %s' % var_perf)
        logger.info('maximum episode return: %s' % max_return)

        # SMAC is minimizing the objective no matter whether run_obj is set to "runtime" or "quality"
        score = - avg_perf # + 0.2 * var_perf - 0.5 * max_return
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
    # nsteps = UniformIntegerHyperparameter("nsteps", 20, 60, default_value=50)
    # lr = UniformFloatHyperparameter("lr", 2e-5, 8e-3, default_value=5e-4)
    units_shared_layer1 = UniformIntegerHyperparameter("units_shared_layer1", 8, 400, default_value=30)
    units_shared_layer2 = UniformIntegerHyperparameter("units_shared_layer2", 8, 400, default_value=100)
    units_policy_layer = UniformIntegerHyperparameter("units_policy_layer", 8, 400, default_value=100)
    vf_coeff = UniformFloatHyperparameter("vf_coeff", 5e-3, 0.6, default_value=0.2)
    ent_coeff = UniformFloatHyperparameter("ent_coeff", 8e-9, 1e-6, default_value=1e-7)
    # gamma = UniformFloatHyperparameter("gamma", 0.7, 0.99, default_value=0.90)
    cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
                             vf_coeff, ent_coeff]) # , gamma, nsteps, lr, ])

    # Create scenario object
    logger.info('Create scenario object')
    logger.info('Output_dir: %s' % smac_output_dir)
    # print(params["run_parallel"])
    if params["run_parallel"].lower() == "true":
        print("RUN PARALLEL")
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

    # Optimize using a smac object:
    seed = np.random.RandomState(params["seed"])
    logger.info('Generate SMAC object')
    smac = SMAC(scenario=scenario, rng=seed, tae_runner=a2c_from_cfg)

    logger.info('Start optimizing algorithm configurations\n')
    optimized_cfg = smac.optimize()

    logger.info('##############################################')
    logger.info('Run training with best configuration again')
    logger.info('##############################################')
    optimized_performance = a2c_from_cfg(optimized_cfg)
    logger.info("Optimized config")
    for k in optimized_cfg:
        logger.info(str(k) + ": " + str(optimized_cfg[k]))
    logger.info("Optimized performance: %.2f" % optimized_performance)

    with open(os.path.join(logdir, 'opt_hyperparams.txt'), 'a') as f:
        for k in optimized_cfg:
            f.write(str(k) + ': ' + str(optimized_cfg[k]) + '\n')
        f.write("Optimized performance: %.2f" % optimized_performance)

    return optimized_cfg


def main():
    args = a2c_arg_parser()
    # print("rhs: " + str(args.run_parallel))
    _ = a2c_smac_wrapper(**args.__dict__)


if __name__ == '__main__':
    main()

