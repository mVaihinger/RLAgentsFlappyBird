import os
import gym
from gym import spaces
from ple import PLE
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt

PIPE_GAP = 150


def process_state(state):
    return np.array(list(state.values()))  # np.array([ .... ])


class PLEEnv_state(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game = getattr(game_module, game_name)(pipe_gap=PIPE_GAP)
        self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.viewer = None

    def step(self, a):
        reward = self.game_state.act(self._action_set[a])
        # reward += 0.1
        state = self.game_state.getGameState()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(
            np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.game_state.getGameState()
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

    # def _update_gravity(self, gravity):  # Either change that or flap power
    #     self.game.set_gravity(gravity)
    #
    # def _update_background_speed(self, speed):
    #     self.game.set_speed(speed)

# class Simulator():
#     def __init__(self, contmean, amplitude, contperiod, discrmean, discrperiod):
#         # Create continuous random line
#         n_screens = options.run_time * 60 * options.speed
#         self.amplitude = options.amplitude
#         self.trace_length = int((n_screens + 1) * options.screenSize[0])
#         point_list = np.stack((np.linspace(0, 1 + n_screens, self.trace_length),
#                                0.5 + np.random.randn(self.trace_length))).T
#
#         # filter random signal
#         filter_order = 4
#         wn = np.array([2, 20], dtype=float) / 1300
#         b, a = spsig.butter(filter_order, wn, btype='bandpass',
#                             analog=False, output='ba')
#
#         point_list[:, 1] = spsig.lfilter(b, a, point_list[:, 1]) + 0.5
#         point_list[:, 1] = [(0.5 + (self.sigmoid(item[1] - 0.5))) for item in point_list]
#         self.point_list = point_list[0::10]

        # Create continuous sinus line

        # Create discrete random line

class Filtered_RandomWalk():
    def __init__(self, trace_length, offset, amplitude, fband):
        self.trace_length = trace_length
        self.points_list = []
        self.filter_order = 4
        self.fband = np.array(fband, dtype=float)
        self.amplitude = amplitude
        self.offset = offset

    def add_random_points(self):
        # Create random data which has mean 0 and std = amplitude
        points = list(np.random.rand(self.trace_length) * self.amplitude - 0.5*self.amplitude)  # Numbers between 0-1,
        b,a = spsig.butter(self.filter_order, self.fband, btype='bandpass', analog=False, output='ba')
        points = list(spsig.lfilter(b,a, points) + self.offset)
        self.points_list = points
        # plt.figure()
        # plt.plot(self.points_list, 'r')
        # plt.show()

class RandomSteps():
    def __init__(self, nsteps, time_interval, value_interval):
        self.nsteps = nsteps
        self.points_list = []
        self.time_interval = time_interval
        self.value_interval = value_interval

    def add_random_points(self):
        # create random switching time points (switch after 30 to 200 environment steps)
        repetitions = list(np.random.randint(low=self.time_interval[0], high=self.time_interval[1], size=self.nsteps))
        # create values of each step
        values = list(np.random.randint(low=self.value_interval[0], high=self.value_interval[1], size=self.nsteps))

        points = []
        for i in range(self.nsteps):
            points += [values[i] for _ in range(repetitions[i])]
        self.points_list = points

        # plt.figure()
        # plt.plot(self.points_list, 'r')
        # plt.show()


class PLEEnv_nonstat(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game = getattr(game_module, game_name)(pipe_gap=PIPE_GAP)
        self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()
        self.random_gravity = None
        # self.simulator = Simulator
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.viewer = None

    def step(self, a):
        reward = self.game_state.act(self._action_set[a])
        # reward += 0.1

        # Set GRAVITY and speed here
        # get value from simulator
        # next_grav, next_speed = self.simulator.get_next_vals()
        # set value in game:
        if len(self.random_speed.points_list) < 1:
            self.random_speed.add_random_points()
        if len(self.random_gravity.points_list) < 1:
            self.random_gravity.add_random_points()
        gravity = self.random_gravity.points_list.pop()
        # print('grav: ' + str(gravity))
        self._update_gravity(gravity)

        background_speed = self.random_speed.points_list.pop()
        # print('speed: ' + str(background_speed))
        self._update_background_speed(background_speed)

        state = self.game_state.getGameState()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(
            np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.game_state.getGameState()
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

    def seed(self, seed=None, **kwargs):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()
        self.random_gravity = Filtered_RandomWalk(trace_length=500, offset=1., amplitude=0.1, fband=[0.0001, 0.005])
        self.random_speed = RandomSteps(nsteps=20, time_interval=[20,200], value_interval=[3,6])  # upper bound is excluded

    # def seed(self, trace_length, offset, amplitude, fband, nsteps, time_interval, value_interval, seed=None):
    #     rng = np.random.RandomState(seed)
    #     self.game_state.rng = rng
    #     self.game_state.game.rng = self.game_state.rng
    #     self.game_state.init()
    #     self.random_gravity = Filtered_RandomWalk(trace_length=trace_length, offset=offset, amplitude=amplitude,
    #                                               fband=fband)
    #     self.random_speed = RandomSteps(nsteps=nsteps, time_interval=time_interval, value_interval=value_interval)  # upper bound is excluded

    def _update_gravity(self, gravity):  # Either change that or flap power
        self.game.set_gravity(gravity)

    def _update_background_speed(self, speed):
        self.game.set_speed(speed)
