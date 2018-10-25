import logging
import numpy as np
import csv
import sys, os, glob
import pickle
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

sys.path.append(os.path.dirname(sys.path[0]))
from A2C.run_a2c import run_a2c_smac
from run_ple_utils import bohb_parser


def a2c_arg_parser():
    parser = bohb_parser()
    parser.add_argument('--nenvs', help='Number of environments', type=int, default=3)

    # TODO comment all variables which shall be optmized. They will be set by the SMAC agent.
    # parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=50,
                        help='number of samples based on which gradient is updated', )
    return parser.parse_args()


def a2c_bohb_wrapper(**params):

    # Setup directories where live data is logged
    logdir = params["logdir"]
    a2c_output_dir = os.path.join(logdir, 'a2c_output')
    # if not os.path.isdir(a2c_output_dir):
    #     os.makedirs(a2c_output_dir)
    params["logdir"] = a2c_output_dir

    bohb_output_dir = os.path.join(logdir, 'bohb_output')
    # if not os.path.isdir(bohb_output_dir):
    #     os.makedirs(bohb_output_dir)

    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
    logger = logging.getLogger()
    logger.propagate = False  # no duplicate logging outputs
    fh = logging.FileHandler(os.path.join(logdir, 'bohb.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)

    # Build configuration space and define all hyperparameters
    cs = ConfigurationSpace()
    lr = UniformFloatHyperparameter("lr", 1e-4, 1e-2, default_value=1e-3)
    units_shared_layer1 = UniformIntegerHyperparameter("units_shared_layer1", 8, 100, default_value=24)
    units_shared_layer2 = UniformIntegerHyperparameter("units_shared_layer2", 8, 100, default_value=24)
    units_policy_layer = UniformIntegerHyperparameter("units_policy_layer", 8, 100, default_value=24)
    vf_coeff = UniformFloatHyperparameter("vf_coeff", 1e-2, 0.5, default_value=0.1)
    ent_coeff = UniformFloatHyperparameter("ent_coeff", 5e-6, 1e-4, default_value=1e-5)
    gamma = UniformFloatHyperparameter("gamma", 0.6, 1., default_value=0.90)
    activ_fcn = CategoricalHyperparameter("activ_fcn", ['relu6', 'elu', 'mixed'], default_value='relu6')
    cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
                            vf_coeff, ent_coeff, gamma, lr, activ_fcn])  # batch_size

    logger.info('##############################################')
    logger.info('Run Optimization')
    logger.info('##############################################')
    if params["array_id"] == 1:
        # Setup directories where live data is logged
        logdir = params["logdir"]
        # a2c_output_dir = os.path.join(logdir, 'a2c_output')
        if not os.path.isdir(a2c_output_dir):
            os.makedirs(a2c_output_dir)
        # params["logdir"] = a2c_output_dir

        # bohb_output_dir = os.path.join(logdir, 'bohb_output')
        if not os.path.isdir(bohb_output_dir):
            os.makedirs(bohb_output_dir)

        # start nameserver
        NS = hpns.NameServer(run_id=params["instance_id"], nic_name=params["nic_name"],
                             working_directory=bohb_output_dir)
        ns_host, ns_port = NS.start()  # stores information for workers to find in working directory

        # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
        worker = A2CWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=params["instance_id"], **params)
        worker.run(background=True)

        # Create scenario object
        logger.info('##############################################')
        logger.info('Setup BOHB instance')
        logger.info('##############################################')

        logger.info('Output_dir: %s' % bohb_output_dir)
        HB = BOHB(configspace=cs,
                  run_id=params["instance_id"],
                  eta=3,
                  min_budget=params["min_resource"],
                  max_budget=params["max_resource"],
                  host=ns_host,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  ping_interval=3600)

        res = HB.run(n_iterations=4,
                     min_n_workers=4)  # BOHB can wait until a minimum number of workers is online before starting

        # pickle result here for later analysis
        with open(os.path.join(bohb_output_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(res, f)

        id2config = res.get_id2config_mapping()
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        # incumbent_trajectory = res.get_incumbent_trajectory()
        # import matplotlib.pyplot as plt
        # plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
        # plt.xlabel('wall clock time [s]')
        # plt.ylabel('incumbent loss')
        # plt.show()

        # shutdown all workers
        HB.shutdown(shutdown_workers=True)

        # shutdown nameserver
        NS.shutdown()

    else:
        host = hpns.nic_name_to_host(params["nic_name"])

        # workers only instantiate the MyWorker, find the nameserver and start serving
        w = A2CWorker(run_id=params["instance_id"], host=host, **params)
        w.load_nameserver_credentials(bohb_output_dir)
        # run worker in the forground,
        w.run(background=False)


class A2CWorker(Worker):

    def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None, **params):
        Worker.__init__(self, run_id=run_id,
                        nameserver=nameserver,
                        nameserver_port=nameserver_port,
                        logger=logger,
                        host=host,
                        id=id)
        self.a2c_logdir = os.path.join(params["logdir"], 'worker{:02d}'.format(params["array_id"]))
        self.params = params

    def compute(self, config, budget, *args, **kwargs):
        dir_list = glob.glob(os.path.join(self.a2c_logdir, 'run*'))
        n_run = len(dir_list)+1
        # while os.path.isdir(os.path.join(self.logdir, 'run{:02d}'.format(n_run))):
        #     # increase run number
        #     n_run += 1
        # rundir = 'run{:02d}'.format(n_run)  # 'run' + str(len(dir_list) + 1)
        os.makedirs(os.path.join(self.a2c_logdir, 'run{:02d}'.format(n_run)))
        self.params["logdir"] = os.path.join(self.a2c_logdir, 'run{:02d}'.format(n_run))

        # print(args.logdir)
        self.params["total_timesteps"] = int(budget)
        avg_perf, var_perf, max_return = run_a2c_smac(**self.params, **config)
        logger = logging.getLogger()
        logger.info('average performance: %s' % avg_perf)
        logger.info('performance variance: %s' % var_perf)
        logger.info('maximum episode return: %s' % max_return)
        # logger.info('interactions till end of run or until convergence: %s' % until_convergence)

        score = - avg_perf  # BOHB is minimizing this.
        logger.info('Quality measure of the current learned agent: %s\n' % score)

        return ({'loss': score,
                 'info': 0})

def main(argv):
    args = a2c_arg_parser()
    # args.worker_id = argv[1]  # Get ID from moab job array to know which job is the first one.
    a2c_bohb_wrapper(**args.__dict__)


if __name__ == '__main__':
    main(1)
    # main(sys.argv)
