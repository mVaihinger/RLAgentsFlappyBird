import os, sys
import gym
import gym.spaces as gspc
import numpy as np

# import gym.Env
# print(sys.path)
# print(os.path.join(os.path.expanduser('~'), 'src', 'PyGame-Learning-Environment'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'src', 'PyGame-Learning-Environment'))
print(sys.path)
from ple.ple import PLE
import time

from envs.random_trajectories import Nonstationarity, load_ns_trajectory

# 18.07 added overlayed sine waves, added right timing
# TODO remove hard coded non-stationarity params


PIPE_GAP = 90


def process_state(state):
    return np.array(list(state.values()))  # np.array([ .... ])


class PLEEnv_state(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, nonstationary=None, nrandfeatures=0, noise_level=0, phase='train'):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game_name = game_name
        self.game = getattr(game_module, game_name)(width=286, pipe_gap=PIPE_GAP)
        self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = gspc.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        if game_name == 'FlappyBird':
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8+nrandfeatures,), dtype=np.float32)
        else:
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(7+nrandfeatures,), dtype=np.float32)
        self.viewer = None

        # nonstationarities
        self.hNS = False
        self.noflap_cnt = 0
        self.flap_cnt = 1
        self.decayed_thrust = 9
        self.nonstationary = nonstationary
        self.phase = phase

        # n rand features
        self.nrandfeat = nrandfeatures

        self.noise = noise_level
        self.noise_gen = self.generate_noise_sources(noise_level)
        # mean divided by the standard deviation.

        def step(a):
            if self.hNS:
                a = self._decay_thrust(a)
            reward = self.game_state.act(self._action_set[a])
            # state = self.game_state.getGameState()
            state = self.get_state()
            state = self.add_noise(state, self.noise)
            # TODO extend state by random uninformative feature(s), do that in all other step functions
            terminal = self.game_state.game_over()

            if self.nonstationary is not None:
                # if self.param_traj.get_len() < 1:
                #     self.param_traj.add_values()
                self.update_param(self.param_traj.get_next_value())

            return state, reward, terminal, {}

        self.step = step

    def _get_image(self):
        image_rotated = np.fliplr(
            np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        if self.game_name == 'FlappyBird':
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8+self.nrandfeat,), dtype=np.float32)
        else:
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(7+self.nrandfeat,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.get_state()
        # state = self.game_state.getGameState()
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def seed(self, seed=None):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()

        offset = self.game.player.GRAVITY
        #TODO: generate 20 long (10M) and short(10000) sequences of random trajectories for training
        #      load trajectory with id=seed in here. If several trajectories are required, load seed, seed+1, tec.
        #TODO: generate 20 long (10M) and short(10000) sequences of random trajectories for testing

        if self.nonstationary == 'hNS':  # nonstationary effect of action
            self.hNS = True
            self.nonstationary = None  # as no dynamics parameter is updated.

            # self.thrust_histry = []
            # self.decay_rate_history = []
        elif self.nonstationary == 'gfNS':  # fast nonstationarity of internal state - > update gravity
            # which parameter?
            self.update_param = self._update_gravity
            self.param_traj = Nonstationarity(self.nonstationary, seed%20, self.phase) # TODO
            # self.param_traj = RandomFloatSteps(nsamples=5000, time_interval=[120, 240],  # 1-2 mins with dt = 500ms
            #                                    value_interval=[0.5, 1.5])  # upper bound is excluded  # TODO set ttrace_length
            # self.param_traj.add_values()
        elif self.nonstationary == 'gsNS':  # slow nonstationarity of internal state - > update gravity
            # which parameter?
            self.update_param = self._update_gravity
            self.param_traj = Nonstationarity(self.nonstationary, seed%20, self.phase) # TODO

            # self.param_traj = Overlayed_RandomSines(nsamples=360000, offset=offset, amplitude=5,
            #                                         fband=[0.000012, 0.000023])
            # self.param_traj.add_values()
        # elif self.nonstationary == 'bfNS':  # fast nonstationarity of effect of action - > update background speed
        #     # which parameter?
        #     self.update_param = self._update_background_speed
        #     self.param_traj = Nonstationarity(self.nonstationary, seed, self.phase) # TODO
        #
        #     # self.param_traj = RandomIntSteps(nsamples=5000, time_interval=[120, 240],  # 1-2 mins with dt = 500ms
        #     #                                  value_interval=[2, 7])  # [5,20] [3,6]upper bound is excluded  # TODO set ttrace_length
        #     # self.param_traj.add_values()

        if self.nrandfeat == 0:
            self.get_state = self.game_state.getGameState
        elif self.nrandfeat > 0:  # random features with same initial statistics than y position of the bird (similar std, mean, step-wise change-rate).
            self.feat_traj = [Nonstationarity('rand_feat', (seed + i) % 20, self.phase) for i in range(self.nrandfeat)]
            # TODO breaks if
            # for traj in self.feat_traj:
            #     traj.add_values()
            self.get_state = self.get_extended_state

    def get_extended_state(self):
        state = self.game_state.getGameState()
        # print(state)
        extended_vals = [traj.get_next_value()*200 for traj in self.feat_traj]
        state = np.concatenate([state, extended_vals])
        # print(state)
        return state

    def generate_noise_sources(self, noise_level):
        feat_val_interval = [350, 17, 200, 200, 50, 200, 200]  # mean value
        feat_val_interval.extend([140] * self.nrandfeat)
        # feat_mean = [200, 1, 100,200,40,100,200]
        # feat_mean.extend([200] * self.nrandfeat)  # 200 for every additional random feature
        noise_std = [1/2 * noise_level * f_int for f_int in feat_val_interval]
        # print(noise_std)
        noise_gen = lambda: [np.random.normal(0, noise_std[i]) for i in range(len(feat_val_interval))]

        return noise_gen

    def add_noise(self, state, noise_level):
        if noise_level == 0:
            return state
        else:
            noise = self.noise_gen()
            state = [s+n for s,n in zip(state, noise)]
            return np.asarray(state)

    def _decay_thrust(self, action):
        if action == 1:  # NO FLAP
            self.noflap_cnt += 1
            # print('NoFlap: {}, {}'.format(self.noflap_cnt, self.flap_cnt))
            # self.decay_rate_history.append(9. * 4/self.flap_cnt)
            self.decayed_thrust -= 9. * 3/min(self.flap_cnt, 300)  # set a lower bound on slope
        elif action == 0:  # FLAP
            self.decayed_thrust = 9
            if not self.noflap_cnt == 0:
                self.noflap_cnt = 0
                self.flap_cnt = 0
            self.flap_cnt += 1
            # self.decay_rate_history.append(0)
            # print('Flap: {}, {}'.format(self.noflap_cnt, self.flap_cnt))

        self.game.player.FLAP_POWER = max(0, min(10, self.decayed_thrust))
        # self.thrust_histry.append(max(0, min(10, self.decayed_thrust)))
        # print(self.game.player.FLAP_POWER)
        return 0

    def _update_gravity(self, gravity):  # Either this or flap power
        self.game.set_gravity(gravity)
        # print(self.game.player.GRAVITY)

    def _update_background_speed(self, speed):
        self.game.set_speed(speed)
        # print(self.game.backdrop.speed)


import matplotlib.pyplot as plt
if __name__ == '__main__':
    p = PLEEnv_state()
    ng = p.generate_noise_sources(0.5)
    noises = []
    for _ in range(200):
        noises.append(ng())
    plt.figure()
    plt.plot(noises)
    plt.show()

    thrust_history = []
    for act in [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,0,0,0]:
        _ = p._decay_thrust(act)
        thrust_history.append(p.game.player.FLAP_POWER)
    plt.figure()
    plt.plot(thrust_history)
    plt.show()

