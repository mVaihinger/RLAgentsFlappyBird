# Implement Actor-Critic architecture for continuous state space, but discrete action space using dense NN as FA

# Based on code of
# - Denny Britz
# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
# -



from ple.games.flappybird import FlappyBird
# from ple.games.pong import Pong

from ple import PLE
import random, time
import numpy as np
import tensorflow as tf
import matplotlib as plt
import pandas as pd
from collections import deque, namedtuple

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

# ===================================================================================================================
#                                                   ACTOR
# ===================================================================================================================
class ActorCritic:
    def __init__(self, sess, env, learning_rate=0.002, epsilon=0.90, epsilon_decay=0.995, discount=0.2, tau=0.125):
        self.sess = sess
        self.env = env

        action_set = env.getActionSet()
        n_actions = len(action_set)
        state_dim = env.getGameState().size
        action_dim = 1

        # Parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.tau = tau

        # Init replay buffer
        self.replay_memory = ReplayBuffer()
        # self.replay_buffer = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        # Init Actor Network and Actor Target Network
        self.actor = Actor(state_dim, n_actions, learning_rate, scope="policy_estimator")
        self.target_actor = Actor(state_dim, n_actions, learning_rate, scope="target_policy_estimator")

        # Init Critic Network and Critic Target Network
        self.critic = Critic(state_dim, action_dim, learning_rate, scope="value_estimator")
        self.target_critic = Critic(state_dim, action_dim, learning_rate, scope="target_value_estimator")  # do we need tau here?



    def learn(self, num_episodes, max_steps_per_episode):
        stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

        env.reset_game()
        done = False

        for idx_eps in range(n_episodes):

            state = self.env.getGameState()
            for idx_step in range(max_steps_per_episode):
                # Choose action epsilon greedy
                action_probs = self.actor.predict_value(sess, [state])
                if random.random() > self.epsilon:
                    action = action_set[np.argmax(action_probs)]
                else:
                    action = random.choice(action_set)
                # Take action
                print(action)
                reward = env.act(action)

                if env.game_over():  # include game over transitions?
                    done = True
                    env.reset_game()
                    break

                # update replay buffer with latest transition
                next_state = env.getGameState()
                self.replay_memory.add_transition(state, action, next_state, reward, done)

                # Update statistics
                stats.episode_rewards[idx_eps] += reward
                stats.episode_lengths[idx_eps] = idx_step

                # Compute target value
                # TD target
                # 1) 1-step TD target
                # TODO predict next_action here and use it for traget computation?
                value_next_state = self.critic.predict_value(sess, [next_state], [[action]])[0]
                target_value_state = reward + self.discount * value_next_state
                # 2) n-step TD target

                # Monte Carlo target (high variance for long horizons/high discount values)
                # 2) empirical return (truncated sum of rewards)
                # 3) empirical return (sum of discounted rewards)

                # Compute TD error = Advantage of the action. Positive if old model underestimates the action value
                td_error = target_value_state - self.critic.predict_value(sess, [state], [[action]])[0]
                # td_error = self.critic.predict_value(sess, [state])[0] - target_value_state #TODO target network or normal one?

                # Update models
                self.critic.update_model(sess, [state], td_error, [[action]])
                self.actor.update_model(sess, [state], self.critic)  # action should be enhanced if td error is positive

                # Print out which step we're on, useful for debugging.
                # print("\rStep {} @ Episode {}/{} ({})".format(
                #     idx_step, idx_eps + 1, num_episodes, stats.episode_rewards[idx_eps - 1]), end="")

                state = next_state

        return stats


# ===================================================================================================================
#                                                   ACTOR
# ===================================================================================================================
class Actor(object):
    """
    Policy Function Approximator

    TODO: How to approximate policy?
    """
    def __init__(self, state_dim, num_actions, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            # Init placeholder
            self.state_in = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name="state")
            self.target_value = tf.placeholder(shape=[None], dtype=tf.float32, name="target")  # Q-value output of critic

            # 2 Layers
            h0_s = tf.layers.dense(self.state_in,
                                   units=20,
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
            h1_s = tf.layers.dense(h0_s,
                                   units=20,
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))

            # Outputs probabilities of possible actions
            self.pred_value = tf.layers.dense(h1_s, num_actions)

            # Compute loss
            self.loss = tf.reduce_mean(-self.target_value)  # TODO loss function
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred_value, self.target_in))

            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

            # Init Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.loss, var_list=self.weights)
            # TODO might need clipping of gradients

    def predict_value(self, sess, states):
        return sess.run(self.pred_value, feed_dict={self.state_in: states})

    def update_model(self, sess, state, target):
        # sess = sess or tf.get_default_session()
        feed_dict = {self.state_in: state, self.target_value: target}
        loss, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
        return loss

# ===================================================================================================================
#                                                   CRITIC
# ===================================================================================================================
class Critic(object):
    """
    Value Function Approximator

    estimates the value of a given state
    updates based on ?? targets..

    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, scope="value_estimator"):
        # Init parameters
        self.learning_rate = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)

        # Init placeholder
        self.state_in = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name="state")
        self.target_value = tf.placeholder(shape=[None], dtype=tf.float32, name="target")

        h0_s = tf.layers.dense(self.state_in,
                               units=20,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
        h1_s = tf.layers.dense(h0_s,
                               units=20,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))

        # augment shape of action_in by one dim to add indices for batching: [0, action],[1, action],...
        self.action_in = tf.placeholder(shape=[None, action_dim], dtype=tf.float32, name="action")
        h0_a = tf.layers.dense(self.action_in,
                               units=20,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.random_uniform_initializer(-1, 1, dtype=tf.float32))

        h2 = tf.layers.dense(tf.concat((h1_s, h0_a), axis=1),
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_uniform_initializer(-1, 1, dtype=tf.float32))
        # h2 = tf.layers.dense(h0,
        #                      units=20,
        #                      activation=tf.nn.relu,
        #                      kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
        # Outputs scalar - the value of the state
        self.pred_value = tf.layers.dense(h2, 1)

        # Compute loss: sum((value - target_value)²)
        self.loss = tf.reduce_mean(tf.square(self.pred_value - self.target_value))
        # self.loss = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.pred_value)

        # For classification:
        # Compute cross-entropy loss per instance: loss = - sum(true_values * log(softmax(pred_values)))
        # Compute total cross-entropy loss: reduce_mean(loss)
        # Or do it all in one step:
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_values, true_values))
        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # Init Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss, var_list=self.weights)
        pass

    def predict_value(self, sess, states, actions):
        """
        Args:
            sess: TensorFlow session
            states: array of states for which we want to predict the actions. [batch_size]
        Return:
            The value prediction of the output tensor. [batch_size, 1]
        """
        return sess.run(self.pred_value, feed_dict={self.state_in: states, self.action_in:actions})

    def update_model(self, sess, states, targets, actions):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
            sess: TensorFlow session.
            states: [current_state] or states of batch
            targets: [current_target] or targets of batch
        """

        feed_dict = {self.state_in: states, self.target_value: targets, self.action_in:actions}
        # evaluate the TF tensors and operations self.loss and self.train_step
        loss, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
        return loss

    def set_optimization_params(self, sess, new_learning_rate):
        """
        Update Learning rate

        Args:
            sess: TensorFlow session
            new_learning_rate: new value of learning rate

        Return:
            The new value of the learning rate
        """
        return sess.run(self.learning_rate, feed_dict={self.learning_rate: new_learning_rate})


# ===================================================================================================================
#                                                   UTILS
# ===================================================================================================================
class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def add_transition(self, state, action, next_state, reward, done):
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


def process_state(state):
    return np.array(list(state.values()))


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


# ===================================================================================================================
#                                                   MAIN
# ===================================================================================================================
if __name__ == "__main__":
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    env.init()
    action_set = env.getActionSet()

    learning_rate = 0.002
    n_episodes = 10000
    max_steps_per_episode = 300

    with tf.Session() as sess:
        actor_critic = ActorCritic(sess, env, learning_rate)
        sess.run(tf.global_variables_initializer())

        # n_actions = len(action_set)
        # state_dim = env.getGameState().size
        #
        # env.init()
        # reward = 0.0
        stats, optimal_policy = actor_critic.learn(n_episodes, max_steps_per_episode)
        plot_episode_stats(stats)


        env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
        for _ in range(100):

            for _ in range(1000):
                if env.game_over():
                    env.reset_game()

                state = env.getGameState()
                env.act(action_set[np.argmax(optimal_policy(sess, [state]))[0]])
