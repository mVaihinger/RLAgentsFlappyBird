import os
import gym
from gym import spaces
from ple import PLE
import numpy as np

PIPE_GAP = 250

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

    def _step(self, a):
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

    def _reset(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.game_state.getGameState()
        return state

    def _render(self, mode='human', close=False):
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

    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()

    # TODO update_env(self, slow_nonstationarity, fast_nonstationarity):
    # game.player.GRAVITY *= 0.5 + random.random()
    # game.player.vel += 40  # game.player.vel sets the y velocity
    # set the velocity of the moving pipes: add function: set_speed() to Pipe class which adapts
    # the value of the pipe speed.