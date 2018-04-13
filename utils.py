import tensorflow as tf
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd


class CategoricalPd:
    def __init__(self, logit):
        self.logit = logit

    def neglogprob(self, x):  # prob that a0 can be described with the probability distribution
        n_cat = self.logit.get_shape().as_list()[-1]
        # returns a vector with length n_act, where one entry with the index x is set to 1 and all other entries
        # are set to 0
        one_hot_actions = tf.one_hot(indices=x, depth=n_cat)
        # compare action probabilities with the one-hot vector
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=one_hot_actions)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logit))
        # add small unifrom noise to distribution --> further exploration
        return tf.argmax(self.logit - tf.log(-tf.log(u)), axis=-1)


class LrDecay:
    def __init__(self, v_init, decay, n_step):
        self.v_init = v_init
        self.decay = decay
        self.step = 0
        self.n_step = n_step

    def value(self):
        v_current = self.v_init * (1 - self.step / self.n_step)  # linear decay
        self.step += 1
        return v_current


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_replay_buffer_size):
        self.max_replay_buffer_size = max_replay_buffer_size
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def add_transition(self, state, action, next_state, reward, done):
        if self.size() > self.max_replay_buffer_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.dones.pop(0)
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    def size(self):
        return len(self._data.states)


def make_epsilon_greedy_policy(predict_fn, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation and epsilon (The probability to select a random action [0,1]) as an
        argument and returns the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = predict_fn(observation, None, None)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig('episode_lengths.png')
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


def process_state(state):
    return np.array(list(state.values()))


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def eval_model(env, model, n_eps):
    action_set = env.getActionSet()
    returns = []
    for _ in range(n_eps):
        env.reset_game()
        total_return = 0.
        while not env.game_over():
            obs = list(env.getGameState())
            action, value, negloprob = model.step([obs])
            reward = env.act(action_set[action[0]])
            total_return += reward
        returns.append(total_return)
    return returns


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for n, (reward, done) in enumerate(zip(rewards[::-1], dones[::-1])):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def lstm(xs, ds, lstm_states, n_lstm=256, scope='lstm'):
    # xs = input states [n_steps, n_in]
    # ds = done flags [n_steps]
    # lstm_states = cell states and hidden states of earlier time steps [n_steps, n_lstm*2]
    n_steps, n_in = xs.get_shape()

    # get LSTM internal states
    c, h = tf.split(tf.squeeze(lstm_states), num_or_size_splits=2, axis=0)  # [n_steps, n_lstm]

    with tf.variable_scope(scope):
        # create weight matrices
        wx = tf.get_variable(name='wx', dtype=tf.float32, shape=[n_in, n_lstm*4])
        wh = tf.get_variable(name='wh', dtype=tf.float32, shape=[n_lstm, n_lstm*4])
        b = tf.get_variable(name='b', dtype=tf.float32, shape=[n_lstm*4])

    # set cell states and hidden states which relate to terminal states to 0
    c = tf.matmul(tf.reshape((1-ds), [-1, 1]), tf.reshape(c, [1, -1]))  # [n_steps, n_lstm]
    h = tf.matmul(tf.reshape((1-ds), [-1, 1]), tf.reshape(h, [1, -1]))  # [n_steps, n_lstm]
    k = tf.matmul(xs, wx) + tf.matmul(h, wh) + b  # [n_steps, n_lstm*4]

    # split into single gates:
    i, f, ca, o = tf.split(k, num_or_size_splits=4, axis=1)
    f = tf.nn.sigmoid(f)  # forget gate ('remember vector') [n_steps, n_lstm]
    i = tf.nn.sigmoid(i)  # input gate ('how much of the input to let into the cell state') [n_steps, n_lstm]
    ca = tf.nn.tanh(ca)   # new candidate values that could be added to the long term memory [n_steps, n_lstm]
    o = tf.nn.sigmoid(o)  # output gate ('focus vector - which information is useful in 'working memory'')
                          # [n_steps, n_lstm]

    # compute new lstm internal states
    c_new = f*c + i*ca        # [n_steps, n_lstm]
    h_new = o*tf.tanh(c_new)  # [n_steps, n_lstm]
    lstm_states_new = tf.concat(values=[c_new, h_new], axis=1)  # [n_steps, n_lstm*2]
    return h_new, lstm_states_new
