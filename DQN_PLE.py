# Deep Q Learning Agent using experience replay and a target network.

import gym
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple

# TODO: implement saving of checkpoints
# TODO: add more statistics values.
# TODO: optimize hyperparameters
# TODO: modify task: i.e. give failure a reward of -10.

CHECKPOINT_DIR = "/media/mara/OS/Users/Mara/Documents/Masterthesis/TrainedModels"

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """

    def __init__(self, num_actions, state_dimension, scope=None):
        self._build_model(num_actions, state_dimension, scope)

    def _build_model(self, num_actions, state_dimension, scope):

        prefix = "target_" if (scope == "target") else ""
        self.state_in = tf.placeholder(shape=[None, state_dimension], dtype=tf.float32, name=prefix+"state_in")  # observations
        self.target_in = tf.placeholder(shape=[None], dtype=tf.float32)  # target Q values
        self.action_in = tf.placeholder(shape=[None, 2], dtype=tf.int32)

        h1 = tf.layers.dense(self.state_in,
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))
        h2 = tf.layers.dense(h1,
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5))

        self.pred_out = tf.layers.dense(h2, num_actions, activation=None, kernel_initializer=None,
                                        name=prefix+"Q_out")  # predicted Q values of each action

        self.loss = tf.losses.mean_squared_error(labels=self.target_in,
                                                 predictions=tf.gather_nd(params=self.pred_out, indices=self.action_in))

        self.optimizer = tf.train.AdamOptimizer(0.0005)
        self.train_step = self.optimizer.minimize(self.loss)

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)

    def predict(self, sess, states):
        """
        Args:
            sess: TensorFlow session
            states: array of states for which we want to predict the actions. [batch_size]
        Return:
            The prediction of the output tensor. [batch_size, n_valid_actions]
        """
        return sess.run(self.pred_out, feed_dict={self.state_in: states})

    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
            sess: TensorFlow session.
            states: [current_state] or states of batch
            actions: [current_action] or actions of batch
            targets: [current_target] or targets of batch
        """
        feed_dict = {self.state_in: states, self.action_in: actions, self.target_in: targets}
        # evaluate the TF tensors and operations self.loss and self.train_step
        loss, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, num_actions, state_dimension, tau=0.001):
        NeuralNetwork.__init__(self, num_actions, state_dimension, scope="target")
        self.tau = tau
        self._counterpart = self._register_counterpart()

    def _register_counterpart(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            # taken from: https://arxiv.org/pdf/1509.02971.pdf
            # theta' = tau * theta + (1 - tau) * theta'
            # where theta' is the parameter of the target network.
            op_holder.append(tf_vars[idx + total_vars // 2].assign((var.value() * self.tau)
                                        + ((1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
        return op_holder

    def update(self, sess):
        for op in self._counterpart:
            sess.run(op)


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


def make_epsilon_greedy_policy(estimator, nA):
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

    def policy_fn(sess, observation, epsilon):

        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(sess, env, estimator, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.4,
               epsilon_decay=.95, use_experience_replay=False, max_replay_buffer_size=4000, batch_size=128,
               target=None, tf_saver=None, save_path=None, save_interval=None):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Implements the options of online learning or using experience replay and also
    target calculation by target networks, depending on the flags. You can reuse
    your Q-learning implementation of the last exercise.

    Args:
        env: PLE game
        approx: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        max_time_per_episode: maximum number of time steps before episode is terminated
        discount_factor: gamma, discount factor of future rewards.
        epsilon: Chance to sample a random action. Float betwen 0 and 1.
        epsilon_decay: decay rate of epsilon parameter
        use_experience_replay: Indicator if experience replay should be used.
        batch_size: Number of samples per batch.
        target: Slowly updated target network to calculate the targets. Ignored if None.

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    replay_buffer = ReplayBuffer(max_replay_buffer_size)
    action_set = env.getActionSet()

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(estimator, len(action_set))

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        avg_reward = np.mean(stats.episode_rewards[max(i_episode-100, 0):i_episode])
        print("\rEpisode {}/{} ({}), avg reward: {}".format(i_episode + 1, num_episodes, last_reward, avg_reward), end="")
        # sys.stdout.flush()

        # Reset the current environment
        env.reset_game()
        state = list(env.getGameState())
        done = False
        loss = None

        # Iterate through steps
        for t in range(max_time_per_episode):
            if env.game_over():
                done = True

            # Update target network maybe
            if target:
                pass

            # Take a step
            action_probs = policy(sess, [state], epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            reward = env.act(action_set[action])
            next_state = list(env.getGameState())

            # episode stats
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            if done:
                print("\rStep {} ({}) loss: {}\n".format(
                    t, max_time_per_episode, loss), end="")
                break

            if use_experience_replay:
                # Update replay buffer
                replay_buffer.add_transition(state, action, next_state, reward, done)

                # Sample minibatch from replay buffer
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
                    replay_buffer.next_batch(min(batch_size, replay_buffer.size()))

                batch_actions = list(zip(range(len(batch_actions)), batch_actions))

                # Calculate TD target for batch. Use "old" fixed parameters if target network is available
                # to compute targets else use "old" parameters of value function estimate.
                batch_next_q_values = (target if target else estimator).predict(sess, batch_next_states)
                batch_best_next_action = np.argmax(batch_next_q_values, axis=1)
                batch_td_target = [batch_rewards[j] + discount_factor *
                                   batch_next_q_values[j][batch_best_next_action[j]] for j in range(len(batch_states))]

                # Update Q value estimator parameters by optimizing between Q network and Q-learning targets
                loss = estimator.update(sess, batch_states, batch_actions, batch_td_target)
            else:
                next_q_values = (target if target else estimator).predict(sess, [next_state])
                best_next_action = np.argmax(next_q_values, axis=1)
                td_target = reward + (discount_factor * next_q_values[0] * best_next_action)
                loss = estimator.update(sess, [state], [[0, action]], td_target)

            if target:
                target.update(sess)

            epsilon *= epsilon_decay
            state = next_state

        if i_episode % save_interval == 0:
            tf_saver.save(sess, save_path, global_step=i_episode)

    return stats


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


if __name__ == "__main__":
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False, state_preprocessor=process_state)

    env.init()

    action_set = env.getActionSet()
    n_actions = len(action_set)
    state_dim = env.getGameState().size

    # Params
    discount = 0.99
    epsilon = 0.4
    epsilon_decay = 0.995
    max_replay_buffer_size = 100000
    batch_size = 128
    max_ep_length = 1000
    num_episodes = 5000
    save_interval = 500

    approx = NeuralNetwork(n_actions, state_dim)
    target = TargetNetwork(n_actions, state_dim)

    saver = tf.train.Saver()
    file_name = "DQN_d" + str(discount) + "_e" + str(epsilon) + "_ed" + str(epsilon_decay) + "_batch" + str(batch_size)
    save_path = os.path.join(CHECKPOINT_DIR, file_name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training
        stats = q_learning(sess, env, approx, num_episodes, max_ep_length,
                           discount_factor=discount,
                           epsilon=epsilon,
                           epsilon_decay=epsilon_decay,
                           use_experience_replay=True,
                           max_replay_buffer_size=max_replay_buffer_size,
                           batch_size=batch_size,
                           target=target,
                           tf_saver=saver,
                           save_path=save_path,
                           save_interval=save_interval)
        # plot_episode_stats(stats)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path + "-" + str(int(round((num_episodes-1)/100, 0) * 100)) + ".meta")
        new_saver.restore(sess, save_path + "-" + str(int(round((num_episodes - 1) / 100, 0) * 100)))
        # new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # graph = tf.get_default_graph()
        # state_in = graph.get_tensor_by_name("state_in:0")
        #
        # # for i in tf.get_default_graph().get_operations():
        # #     print(i.name)
        #
        # var_list = tf.get_collection(tf.GraphKeys.TRAIN_OP)
        # print(var_list)
        #
        # action_pred = graph.get_operation_by_name("pred_out")

        # Evaluation of trained model
        env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
        for _ in range(100):

            for _ in range(1000):
                # Update Player parameters gravity and horizontal speeds
                #     # if i % 100 == 0:
                #         # game.player.GRAVITY *= 0.5 + random.random()
                #         # game.player.vel += 40
                #  game.player.vel only sets the y velocity. But I want to set the background velocity - TODO how??
                if env.game_over():
                    env.reset_game()

                state = env.getGameState()
                # env.act(action_set[np.argmax(sess.run(action_pred, {state_in: state}))])
                env.act(action_set[np.argmax(approx.predict(sess, [state]))])
