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
from DQN.run_dqn import run_dqn_smac
from run_ple_utils import bohb_parser


def dqn_arg_parser():
    parser = bohb_parser()
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


def dqn_bohb_wrapper(**params):

    # Setup directories where live data is logged
    logdir = params["logdir"]
    dqn_output_dir = os.path.join(logdir, 'dqn_output')
    # if not os.path.isdir(dqn_output_dir):
    #     os.makedirs(dqn_output_dir)
    params["logdir"] = dqn_output_dir

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
    epsilon = UniformFloatHyperparameter("epsilon", 0.2, 0.9, default_value=0.6)  # initial epsilon
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

    logger.info('##############################################')
    logger.info('Run Optimization')
    logger.info('##############################################')
    if params["array_id"] == 1:
        # Setup directories where live data is logged
        # logdir = params["logdir"]
        # dqn_output_dir = os.path.join(logdir, 'dqn_output')
        if not os.path.isdir(dqn_output_dir):
            os.makedirs(dqn_output_dir)
        # params["logdir"] = dqn_output_dir

        # bohb_output_dir = os.path.join(logdir, 'bohb_output')
        if not os.path.isdir(bohb_output_dir):
            os.makedirs(bohb_output_dir)

        # start nameserver
        NS = hpns.NameServer(run_id=params["instance_id"], nic_name=params["nic_name"],
                             working_directory=bohb_output_dir)
        ns_host, ns_port = NS.start()  # stores information for workers to find in working directory

        # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
        worker = DQNWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=params["instance_id"], **params)
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
        w = DQNWorker(run_id=params["instance_id"], host=host, **params)
        w.load_nameserver_credentials(bohb_output_dir)
        # run worker in the forground,
        w.run(background=False)


class DQNWorker(Worker):

    def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None, **params):
        Worker.__init__(self, run_id=run_id,
                        nameserver=nameserver,
                        nameserver_port=nameserver_port,
                        logger=logger,
                        host=host,
                        id=id)
        self.logdir = os.path.join(params["logdir"], 'worker{:02d}'.format(params["array_id"]))
        self.params = params

    def compute(self, config, budget, *args, **kwargs):
        dir_list = glob.glob(os.path.join(self.logdir, 'run*'))
        n_run = len(dir_list)+1
        # while os.path.isdir(os.path.join(self.logdir, 'run{:02d}'.format(n_run))):
        #     # increase run number
        #     n_run += 1
        # rundir = 'run{:02d}'.format(n_run)  # 'run' + str(len(dir_list) + 1)
        os.makedirs(os.path.join(self.logdir, 'run{:02d}'.format(n_run)))
        self.params["logdir"] = os.path.join(self.logdir, 'run{:02d}'.format(n_run))

        # print(args.logdir)
        self.params["total_timesteps"] = int(budget)
        avg_perf, var_perf, max_return = run_dqn_smac(**self.params, **config)
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
    args = dqn_arg_parser()
    # args.worker_id = argv[1]  # Get ID from moab job array to know which job is the first one.
    dqn_bohb_wrapper(**args.__dict__)


if __name__ == '__main__':
    main(1)
    # main(sys.argv)
