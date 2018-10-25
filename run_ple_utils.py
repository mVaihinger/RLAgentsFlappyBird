"""
Helpers for scripts like run_ple.py.
"""

import os
import json, time, csv
import gym
import numpy as np

# import logger
import envs.environment
from envs.vec_env import SubprocVecEnv
from utils_OAI import set_global_seeds


def make_ple_envs(env_id, num_env, seed, start_index=0, *args, **kwargs):
    """
    Create a monitored SubprocVecEnv for PLE.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank, *args, **kwargs) # TODO should be after the monitor command!
            env = Monitor(env, None, **kwargs)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), **kwargs)
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], env_id)

def make_ple_env(env_id, seed, **kwargs):
    env = gym.make(env_id)
    env.seed(seed, **kwargs)
    set_global_seeds(seed)
    return env


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--architecture', help='Policy architecture', choices=['ff', 'lstm', 'gru'],
                        default='ff')
    # parser.add_argument('--activ_fcn', choices=['relu6', 'elu', 'mixed'], type=str, default='relu6',
    #                     help='Activation functions of network layers', )

    parser.add_argument('--env', help='environment ID', default='ContFlappyBird-v1')
    parser.add_argument('--test_env', help='testv environment ID', default='ContFlappyBird-v3')
    parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(2e4))
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    # TODO if lr and gamma are optimized, remove them here!
    # parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.01,
                        help='Maximum gradient norm up to which gradient is not clipped', )
    # parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.95)
    # parser.add_argument('--batch_size', type=int, default=50,
    #                     help='number of samples based on which gradient is updated', )

    parser.add_argument('--log_interval', type=int, default=20,
                        help='Network parameter values are stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ')
    parser.add_argument('--show_interval', type=int, default=1,
                        help='Env is rendered every n-th episode. 0 = no rendering')
    parser.add_argument('--logdir', default='/home/mara/Desktop/logs/A2C_OAI_NENVS',
                        help='directory where logs are stored')
    parser.add_argument('--eval_model', choices=['all', 'inter', 'final'], default='inter',
                        help='Eval all stored models, only the final model or only the intermediately stored models (while testing the best algorithm configs)')
    parser.add_argument('--keep_model', help='How many best models shall be kept during training. 0 -> only final model', type=int, default=2)

    # For configuration validation.
    parser.add_argument('--test_interval', type=int, default=0,
                        help='Model is evaluated after <test_interval> model updates. 0 = do not evaluate while learning.')
    return parser

def smac_parser():
    parser = arg_parser()
    parser.add_argument('--runcount_limit', type=int, default=int(4),
                        help='amount of algorithm evaluations allowed to optimize hyperparameters')
    parser.add_argument('--run_parallel', choices=["True", "true", "False", "false"], type=str, default="false",
                        help='flag which determines whether smac instances are run in parallel or not.')
    parser.add_argument('--instance_id', help='id of the smac instance', type=int, default=3)
    parser.add_argument('--early_stop', help='stop bad performing runs ealier', type=bool, default=True)
    return parser

def bohb_parser():
    parser = arg_parser()
    parser.add_argument('--instance_id', help='unique id to identify the HPB run. id = [0, s_max]', type=str, default=1)
    parser.add_argument('--array_id', help='resource manager array id to treat one job as a HPB run.', default=1, type=int)
    parser.add_argument('--nic_name', help='name of the Network Interface Card', default='boot0', type=str)
    parser.add_argument('--min_resource', help='minimum resource in terms of interactions with the env.', default=32, type=int)
    parser.add_argument('--max_resource', help='maximum resource in terms of interactions with the env.', default=512, type=int)
    parser.add_argument('--early_stop', help='stop bad performing runs ealier', type=bool, default=False)
    # parser.add_argument('--host', help='Host Ip adress', type=str)
    return parser

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
                        print('Argument %s is out of bounds. Value is %s. Should be in %s. Set default.' % (k, v, self.dict[k][:-2]))
                else:
                    print('Argument %s doesn\'t have expected data type %s. Set default.' % (k, self.dict[k][-1]))
        return params

def params_parser():
    paramDict = ParamDict()
    paramDict.add_cat_param("architecture", options=['ff', 'lstm', 'gru'], default='ff', dtype=str)
    paramDict.add_cat_param("activ_fcn", options=['relu6', 'elu', 'mixed'], default='relu6', dtype=str)
    paramDict.add_cat_param("env", options=['FlappyBird-v1', 'FlappyBird-v2', 'FlappyBird-v3', 'FlappyBird-v4', 'ContFlappyBird-v1', 'ContFlappyBird-v2', 'ContFlappyBird-v3', 'ContFlappyBird-v4'], default='FlappyBird-v1', dtype=str)
    paramDict.add_cat_param("test_env", options=['FlappyBird-v1', 'FlappyBird-v2', 'FlappyBird-v3', 'FlappyBird-v4', 'ContFlappyBird-v1', 'ContFlappyBird-v2', 'ContFlappyBird-v3', 'ContFlappyBird-v4'], default='FlappyBird-v1', dtype=str)
    paramDict.add_num_param("total_timesteps", lb=0, ub=10e15, default=int(10e3), dtype=int)
    paramDict.add_num_param("seed", lb=0, ub=np.inf, default=123, dtype=int)

    paramDict.add_num_param("lr", lb=1e-12, ub=1., default=5e-4, dtype=float)
    paramDict.add_num_param("max_grad_norm", lb=0.001, ub=20, default=0.01, dtype=float)
    paramDict.add_num_param("gamma", lb=0.01, ub=1., default=0.90, dtype=float)
    paramDict.add_num_param("batch_size", lb=1, ub=500, default=50, dtype=int)

    paramDict.add_num_param("log_interval", lb=0, ub=1e7, default=100, dtype=int)
    paramDict.add_num_param("show_interval", lb=0, ub=1e7, default=0, dtype=int)
    paramDict.add_cat_param("logdir", options=None, default='/home/mara/Desktop/logs/A2C_OAI_NENVS', dtype=str)
    paramDict.add_cat_param("eval_model", options=['all', 'final', 'inter'], default='all', dtype=str)
    paramDict.add_cat_param("early_stop", options=[True, False], default=False, dtype=bool)
    paramDict.add_num_param("keep_model", lb=0, ub=500, default=2, dtype=int)

    # For configuration validation.
    paramDict.add_num_param("test_interval", lb=0, ub=1e7, default=0, dtype=int)

    return paramDict  # .check_params(**kwargs)


# def arg_parser():
#     """
#     Create an empty argparse.ArgumentParser.
#     """
#     import argparse
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--env', help='environment ID', default='FlappyBird-v1')
#     parser.add_argument('--seed', help='RNG seed', type=int, default=0)
#     parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(10e5))
#     return parser


from gym.core import Wrapper

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, 'env_id' : env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+reset_keywords+info_keywords)
            self.logger.writeheader()
            self.f.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass

# def get_monitor_files(dir):
#     return glob(os.path.join(dir, "*" + Monitor.EXT))
#
# def load_results(dir):
#     import pandas
#     monitor_files = (
#         glob(os.path.join(dir, "*monitor.json")) +
#         glob(os.path.join(dir, "*monitor.csv"))) # get both csv and (old) json files
#     if not monitor_files:
#         raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
#     dfs = []
#     headers = []
#     for fname in monitor_files:
#         with open(fname, 'rt') as fh:
#             if fname.endswith('csv'):
#                 firstline = fh.readline()
#                 assert firstline[0] == '#'
#                 header = json.loads(firstline[1:])
#                 df = pandas.read_csv(fh, index_col=None)
#                 headers.append(header)
#             elif fname.endswith('json'): # Deprecated json format
#                 episodes = []
#                 lines = fh.readlines()
#                 header = json.loads(lines[0])
#                 headers.append(header)
#                 for line in lines[1:]:
#                     episode = json.loads(line)
#                     episodes.append(episode)
#                 df = pandas.DataFrame(episodes)
#             else:
#                 assert 0, 'unreachable'
#             df['t'] += header['t_start']
#         dfs.append(df)
#     df = pandas.concat(dfs)
#     df.sort_values('t', inplace=True)
#     df.reset_index(inplace=True)
#     df['t'] -= min(header['t_start'] for header in headers)
#     df.headers = headers # HACK to preserve backwards compatibility
#     return df
