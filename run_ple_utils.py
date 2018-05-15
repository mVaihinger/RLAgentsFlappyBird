"""
Helpers for scripts like run_ple.py.
"""

import os
import json, time, csv
import gym

import logger
import envs.environment
from envs.vec_env import SubprocVecEnv
from utils_OAI import set_global_seeds


def make_ple_envs(env_id, num_env, seed, start_index=0):
    """
    Create a monitored SubprocVecEnv for PLE.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_ple_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    set_global_seeds(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='FlappyBird-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser


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
