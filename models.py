import tensorflow as tf
import numpy as np

from utils import CategoricalPd
from utils import lstm

# TODO add weights and biases initializer

class SharedMLP:
    def __init__(self, sess, state_dim, n_actions, reuse=False):
        # Model Input
        self.obs_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='obs_in')
        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.layers.dense(self.obs_in, units=20, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, units=20, activation=tf.nn.relu)

            self.ap_out = tf.layers.dense(h2, units=n_actions, activation=None)  # action probabilities
            self.vf_out = tf.layers.dense(h2, units=1, activation=None)  # state value

        # The output of the NN are non-normalized action probabilities. They are converted to a probabiltiy
        # distribution from which normalized probabilities can be sampled.
        self.pd = CategoricalPd(self.ap_out)  # Init the distribution with output values of NN
        a0 = self.pd.sample()  # sample probabilities for each action from probability distribution which adds small unifrom noise to the prob distribution derived from NN output (a0=[n_actions])
        v0 = self.vf_out[:, 0]

        neglogprob0 = self.pd.neglogprob(a0)  # a0 are the labels for the cross entropy computation
        self.initial_states = None

        # Prediction functions for a complete step and for the state value only
        def step(obs, dones, lstm_states):
            a, v, neglogprob = sess.run([a0, v0, neglogprob0], {self.obs_in: obs})
            return a, v, self.initial_states, neglogprob

        def value(obs, dones, lstm_states):
            return sess.run(v0, {self.obs_in: obs})
            # return sess.run(self.vf_out, {self.obs_in: obs})

        self.step = step
        self.value = value
        self.a0 = a0


class LSTM_CatPD:
    def __init__(self, sess, state_dim, n_actions, n_steps, n_lstm=256, reuse=False):
        self.obs_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='obs_in') # observations
        self.D = tf.placeholder(dtype=tf.float32, shape=[None], name='dones')  # dones
        self.LS = tf.placeholder(dtype=tf.float32, shape=[None, n_lstm*2], name='lstm_s')  # cell and hidden states

        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.layers.dense(self.obs_in, units=20, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, units=20, activation=tf.nn.relu)

            # LSTM cell
            h3, s_new = lstm(h2, self.D, self.LS, scope='lstm', n_lstm=n_lstm)

            self.ap_out = tf.layers.dense(h3, units=n_actions, activation=None)
            self.vf_out = tf.layers.dense(h3, units=1, activation=None)

        # The output of the NN are non-normalized action probabilities. They are converted to a probabiltiy
        # distribution from which normalized probabilities can be sampled.
        self.pd = CategoricalPd(self.ap_out)  # Init the distribution with output values of NN
        a0 = self.pd.sample()  # sample probabilities for each action from probability distribution which adds small unifrom noise to the prob distribution derived from NN output (a0=[n_actions])
        v0 = self.vf_out[:, 0]

        neglogprob0 = self.pd.neglogprob(a0)  # a0 are the labels for the cross entropy computation
        self.initial_states = [np.zeros(shape=n_lstm*2, dtype=np.float32)]

        def step(obs, dones, lstm_states):
            return sess.run([a0, self.ap_out, v0, s_new, neglogprob0], {self.obs_in: obs, self.D: dones, self.LS: lstm_states})

        def value(obs, dones, lstm_states):
            return sess.run(v0, {self.obs_in: obs, self.D: dones, self.LS: lstm_states})
            # return sess.run([self.vf_out], {self.obs_in: obs, self.D: dones, self.LS: lstm_states})

        self.step = step
        self.value = value
        self.a0 = a0

class LSTM_SM:
    def __init__(self, sess, state_dim, n_actions, n_steps, n_lstm=256, reuse=False):
        self.obs_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='obs_in') # observations
        self.D = tf.placeholder(dtype=tf.float32, shape=[None], name='dones')  # dones
        self.LS = tf.placeholder(dtype=tf.float32, shape=[None, n_lstm*2], name='lstm_s')  # cell and hidden states

        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.layers.dense(self.obs_in, units=20, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, units=20, activation=tf.nn.relu)

            # LSTM cell
            h3, s_new = lstm(h2, self.D, self.LS, scope='lstm', n_lstm=n_lstm)

            self.ap_out = tf.layers.dense(h3, units=n_actions, activation=tf.nn.softmax)
            self.vf_out = tf.layers.dense(h3, units=1, activation=None)

        # The output of the NN are non-normalized action probabilities. They are converted to a probabiltiy
        # distribution from which normalized probabilities can be sampled.
        # self.aps = tf.squeeze(tf.nn.softmax(self.ap_out))
        # a0 = np.random.choice(np.arange(n_actions), p=self.ap_out)
        v0 = self.vf_out[:, 0]
        # picked_action_prob = tf.gather(self.ap_out, a0)  # a0 are the labels for the cross entropy computation

        self.initial_states = [np.zeros(shape=n_lstm*2, dtype=np.float32)]

        def step(obs, dones, lstm_states):
            return sess.run([self.ap_out, v0, s_new], {self.obs_in: obs, self.D: dones, self.LS: lstm_states})
            # return sess.run([a0, self.ap_out, v0, s_new, picked_action_prob], {self.obs_in: obs, self.D: dones, self.LS: lstm_states})

        def value(obs, dones, lstm_states):
            return sess.run(v0, {self.obs_in: obs, self.D: dones, self.LS: lstm_states})
            # return sess.run([self.vf_out], {self.obs_in: obs, self.D: dones, self.LS: lstm_states})

        self.step = step
        self.value = value
        # self.a0 = a0

class DQN():
    """
    Deep Q Network class based on TensorFlow.
    """
    def __init__(self, sess, state_dimension, num_actions, scope=None, reuse=False):
        prefix = "target_" if (scope == "target") else ""
        self.obs_in = tf.placeholder(shape=[None, state_dimension], dtype=tf.float32,
                                       name=prefix + "state_in")  # observations

        # Network Architecture
        # with tf.variable_scope(scope, reuse=reuse): # leads to error when assigning weights to target network
        h1 = tf.layers.dense(self.obs_in,
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
        h2 = tf.layers.dense(h1,
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
        # Output: predicted Q values of each action
        self.pred_out = tf.layers.dense(h2, num_actions, activation=None, kernel_initializer=None)  #,
                                        #name=prefix + "Q_out")  # predicted Q values of each action

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)

        def predict(obs, dones, lstm_states):
            """
            Args:
                sess: TensorFlow session
                obs: array of observatons for which we want to predict the actions. [batch_size]
            Return:
                The prediction of the output tensor. [batch_size, n_valid_actions]
            """
            return sess.run(self.pred_out, feed_dict={self.obs_in: obs})

        self.predict = predict


class MLP:
    def __init__(self, sess, state_dim, n_actions, reuse=False):
        # Model Input
        self.obs_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='obs_in')
        with tf.variable_scope("model", reuse=reuse):
            # Policy Network
            h1 = tf.layers.dense(self.obs_in, units=20, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, units=20, activation=tf.nn.relu)
            self.ap_out = tf.layers.dense(h2, units=n_actions, activation=None)  # action probabilities

            # Value Network
            h3 = tf.layers.dense(self.obs_in, units=20, activation=tf.nn.relu)
            h4 = tf.layers.dense(h1, units=20, activation=tf.nn.relu)
            self.vf_out = tf.layers.dense(h2, units=1, activation=None)  # state value

        # The output of the NN are non-normalized action probabilities. They are converted to a probabiltiy
        # distribution from which normalized probabilities can be sampled.
        self.pd = CategoricalPd(self.ap_out)  # Init the distribution with output values of NN
        a0 = self.pd.sample()  # sample probabilities for each action from probability distribution which adds small unifrom noise to the prob distribution derived from NN output (a0=[n_actions])
        v0 = self.vf_out[:, 0]

        neglogprob0 = self.pd.neglogprob(a0)  # a0 are the labels for the cross entropy computation
        self.initial_states = None

        # Prediction functions for a complete step and for the state value only
        def step(obs, dones, lstm_states):
            a, v, neglogprob = sess.run([a0, v0, neglogprob0], {self.obs_in: obs})
            return a, v, self.initial_states, neglogprob

        def value(obs, dones, lstm_states):
            return sess.run(v0, {self.obs_in: obs})
            # return sess.run(self.vf_out, {self.obs_in: obs})

        self.step = step
        self.value = value
        self.a0 = a0
