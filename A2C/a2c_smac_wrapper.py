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
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'RLAgentsFlappyBird'))
from run_a2c import run_a2c_smac

# TODO ##############################
# TODO NEXT LEVEL: Cluster :D
# TODO rsync everything to the cluster
# TODO check whether normal run works

# check whether parallel run works

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nenvs', help='Number of environments', type=int, default=3)
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'casta', 'largemlp'],
                        default='largemlp')
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float,
                        default=0.01)
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.95)
    parser.add_argument('--log_interval',
                        help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ',
                        type=int, default=30)
    # parser.add_argument('--save_interval', help='Model is saved after <save_interval> model updates', type=int,
    #                     default=1000)
    parser.add_argument('--show_interval', help='Env is rendered every n-th episode. 0 = no rendering', type=int,
                        default=0)
    parser.add_argument('--logdir', help='directory where logs are stored',
                        default='/home/mara/Desktop/logs/A2C_OAI_NENVS')  # '/mnt/logs/A2C')
    parser.add_argument('--seed', help='RNG seed', type=int, default=2)
    parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(5e3))
    parser.add_argument('--runcount_limit', help='amount of algorithm evaluations allowed to optimize hyperparameters',
                        type=int, default=int(3))
    parser.add_argument('--eval_model', help='eval all stored models or only final model', choices=['all', 'final'],
                        default='all')
    parser.add_argument('--run_parallel', help='flag which determine whethe smac instances are run in parallel or not.', choices=["True", "true", "False", "false"], type=str, default="false")
    parser.add_argument('--instance_id', help='id of the smac instance', type=int, default=1)
    args = parser.parse_args()
    return args

def a2c_smac_wrapper(**params):
    # logdir = os.path.join(params["logdir"], str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # os.makedirs(logdir)
    logdir = params["logdir"]

    a2c_output_dir = os.path.join(logdir, 'a2c_output' + str(params["instance_id"]))
    if not os.path.isdir(a2c_output_dir):  # TODO check what this does
        os.makedirs(a2c_output_dir)
    smac_output_dir = os.path.join(logdir, 'smac3_output' + str(params["instance_id"]))
        # params["seed"] = id  # int(time.time() * 1000)

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

        score = - (avg_perf - 0.2 * var_perf + 0.5 * max_return)  # SMAC is minimizing this.
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
    nsteps = UniformIntegerHyperparameter("nsteps", 20, 60, default_value=50)
    lr = UniformFloatHyperparameter("lr", 2e-5, 8e-3, default_value=5e-4)
    units_shared_layer1 = UniformIntegerHyperparameter("units_shared_layer1", 8, 256, default_value=64)
    units_shared_layer2 = UniformIntegerHyperparameter("units_shared_layer2", 8, 256, default_value=64)
    units_policy_layer = UniformIntegerHyperparameter("units_policy_layer", 8, 256, default_value=64)
    vf_coeff = UniformFloatHyperparameter("vf_coeff", 1e-2, 0.4, default_value=0.2)
    ent_coeff = UniformFloatHyperparameter("ent_coeff", 1e-9, 1e-6, default_value=1e-7)
    # gamma = UniformFloatHyperparameter("gamma", 0.7, 0.99, default_value=0.90)
    cs.add_hyperparameters([nsteps, lr, units_shared_layer1, units_shared_layer2, units_policy_layer,
                             vf_coeff, ent_coeff]) # , gamma])

    # Create scenario object
    logger.info('Create scenario object')
    logger.info('Output_dir: %s' % smac_output_dir)
    # print(params["run_parallel"])
    if params["run_parallel"].lower() == "true":
        print("RUNNNNNNN PARALLEL!!!!!!!!!!!!!!")
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

if __name__ == '__main__':
    args = arg_parser()

    logdir = os.path.join(args.logdir, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(logdir)
    args.logdir = logdir
    optimized_cfg = a2c_smac_wrapper(**args.__dict__)