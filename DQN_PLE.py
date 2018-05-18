# Deep Q Learning Agent using experience replay and a target network.

# import gym
# from ple.games.flappybird import FlappyBird
# from ple import PLE
import numpy as np
import sys, os, datetime
import tensorflow as tf
# from collections import namedtuple

from models_OAI import DQN
from utils_OAI import ReplayBuffer
from utils_OAI import make_epsilon_greedy_policy, normalize_obs
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session, set_global_seeds

# TODO: implement saving of checkpoints
# TODO: add more statistics values.
# TODO: optimize hyperparameters
# TODO: Prioritize experience in replay buffer
# TODO check initital state
# TODO add replay start value
# summary doesn't work for network params
# How to add target network?
# How to add duelling network?


CHECKPOINT_DIR = "/media/mara/OS/Users/Mara/Documents/Masterthesis/TrainedModels"
DATE = str(datetime.datetime.today())
LOG_FILE = os.path.join('/home/mara/Desktop/logs/DQN', DATE)

# EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])
EpisodeStats = {"episode_lengths": [0],
                "episode_rewards": [0]}


class DQNAgent():
    """
    Neural Network class based on TensorFlow.
    """

    def __init__(self, ob_space, ac_space, scope="model", lr=0.0005, reuse=False, log_interval=0, write_summary=False):
        sess = make_session()  # TODO add CPU config information

        self.num_steps_trained = 0
        # self.log_interval = log_interval
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)

        # Targets in loss computation
        self.target_in = tf.placeholder(shape=[None], dtype=tf.float32)  # target Q values
        self.action_in = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.nbatch =tf.Variable(initial_value=0, trainable=False)
        # nbatch = self.target_in.shape[1].value if (self.target_in.shape[0].value is not None) else 0

        train_model = DQN(sess, ob_space, ac_space.n, scope, reuse=reuse)
        # target_model = TargetNetwork(sess, state_dimension, num_actions)

        loss = tf.losses.mean_squared_error(labels=self.target_in,
                                            predictions=tf.gather_nd(params=train_model.pred_out,
                                                                          indices=self.action_in))
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(lr)
        # self.train_step = self.optimizer.minimize(self.loss)
        grads = optimizer.compute_gradients(loss)
        _train = [optimizer.apply_gradients(grads),
                  self.global_step.assign_add(self.nbatch)]

        if write_summary:
            for g, v in grads:
                if g is not None:
                    tf.summary.histogram("train/grads/%s-grad" % v.name, g)
            for p in params:
                if p is not None:
                    tf.summary.histogram("train/params/%s" % p.name, p.value())
            tf.summary.scalar("train/vf_loss", loss)
            self.summary_step = tf.summary.merge_all()

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)

        def train(obs, actions, targets):
            """
            Updates the weights of the neural network, based on its targets, its
            predictions, its loss and its optimizer.

            Args:
                sess: TensorFlow session.
                obs: [current_observation] or observations of batch
                actions: [current_action] or actions of batch
                targets: [current_target] or targets of batch
            """
            feed_dict = {train_model.obs_in: obs, self.action_in: actions, self.target_in: targets, self.nbatch: len(targets)}
            # evaluate the TF tensors and operations self.loss and self.train_step
            total_loss, _, global_step = sess.run([loss, _train, self.global_step], feed_dict=feed_dict)

            if write_summary and (log_interval != 0) and (self.num_steps_trained % log_interval == 0.):
                summary_str = sess.run(self.summary_step, feed_dict)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)

            self.num_steps_trained += 1

            return total_loss

        self.train_model = train_model
        # self.target_model = target_model
        self.train = train
        self.predict = train_model.predict
        # self.save = save_params
        # self.load = load_params
        tf.global_variables_initializer().run(session=sess)

        # Set the logs writer to the folder /tmp/tensorflow_logs
        self.summary_writer = tf.summary.FileWriter(LOG_FILE, graph_def=sess.graph_def)

    def get_summary_writer(self):
        return self.summary_writer



class TargetNetwork(DQN):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, sess, state_dimension, num_actions, tau=0.001):
        self. sess = sess
        DQN.__init__(self, sess, state_dimension, num_actions, scope="model", reuse=True)
        self.tau = tau
        self._counterpart = self._register_counterpart()

        def update():
            for op in self._counterpart:
                self.sess.run(op)

        self.update = update
        tf.global_variables_initializer().run(session=sess)

    def _register_counterpart(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        # (total_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            # taken from: https://arxiv.org/pdf/1509.02971.pdf
            # theta' = tau * theta + (1 - tau) * theta'
            # where theta' is the parameter of the target network.
            print(var)
            print(var.value())
            op_holder.append(tf_vars[idx + total_vars // 2].assign((var.value() * self.tau)
                                        + ((1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
        return op_holder


def q_learning(env, seed, total_timesteps=int(1e8), gamma=0.95, epsilon=0.4, epsilon_decay=.95,
               use_experience_replay=False, max_replay_buffer_size=4000, batch_size=128,
               lr=7e-4, lrschedule='linear', target=None, log_interval= 100,
               tf_saver=None, save_path=None, save_interval=None, write_summary=False):
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
    tf.reset_default_graph()
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space

    model = agent = DQNAgent(ob_space, ac_space, log_interval=log_interval, write_summary=write_summary)
    if write_summary:
        summary_writer = model.get_summary_writer()

    # Keeps track of useful statistics
    stats = EpisodeStats

    replay_buffer = ReplayBuffer(max_replay_buffer_size)
    # action_set = env.getActionSet()


    # START LEARNING
    i_episode = 0
    done = False
    # loss = None

    # Reset env
    obs = np.zeros(ob_space.shape, dtype=np.float32)
    next_obs = env.reset()
    # state = list(env.getGameState())

    # The policy we're following
    policy = make_epsilon_greedy_policy(agent.predict, ac_space.n)

    for update in range(1, total_timesteps):

        if done:  # env.game_over():
            # done = True
            # Reset the current environment
            obs = env.reset()
            obs = normalize_obs(obs)

            if write_summary:
                summary = tf.Summary()
                summary.value.add(tag='env/ep_return', simple_value=stats['episode_rewards'][i_episode])
                summary.value.add(tag="env/ep_length", simple_value=stats['episode_lengths'][i_episode])
                summary_writer.add_summary(summary, i_episode)
                summary_writer.flush()
            i_episode += 1
            stats['episode_lengths'].append(0)
            stats['episode_rewards'].append(0)

        # if i_episode > 0:
            # Print out which episode we're on, useful for debugging.
            # Also print return for last episode
            # last_reward = stats['episode_rewards'][i_episode]  # stats.episode_rewards[i_episode - 1]
            # avg_reward = np.mean(stats['episode_rewards'][max(i_episode-100, 0):i_episode])
            # print("\rEpisode {} ({}), avg reward: {}".format(i_episode + 1, last_reward, avg_reward), end="")
            # sys.stdout.flush()

        # render only every i-th episode
        if log_interval != 0:
            if i_episode % log_interval == 0:
                env.render()

        # Update target network
        if target:
            target.update()

        # Take a step
        print(agent.predict([obs], None, None))
        action_probs = policy([obs], epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_obs, reward, done, _ = env.step(action)  # state is not resettet here..
        reward -= 1e-5
        next_obs = normalize_obs(next_obs)

        # reward = env.act(action_set[action])
        #
        # next_state = list(env.getGameState())

        # episode stats
        stats['episode_lengths'][i_episode] += 1
        stats['episode_rewards'][i_episode] += reward
        # print(i_episode)
        # print(stats['episode_lengths'], stats['episode_rewards'])


        if use_experience_replay:
            # Update replay buffer
            replay_buffer.add_transition(obs, action, next_obs, reward, done)

            # Sample minibatch from replay buffer
            batch_states, batch_actions, batch_next_states, batch_rewards, _, batch_dones = \
                replay_buffer.next_batch(min(batch_size, replay_buffer.size()))

            batch_actions = list(zip(range(len(batch_actions)), batch_actions))

            # Calculate TD target for batch. Use "old" fixed parameters if target network is available
            # to compute targets else use "old" parameters of value function estimate.
            batch_next_q_values = (target if target else agent.train_model).predict(batch_next_states, None, None)
            batch_best_next_action = np.argmax(batch_next_q_values, axis=1)
            batch_td_target = [batch_rewards[j] + gamma *
                               batch_next_q_values[j][batch_best_next_action[j]] for j in range(len(batch_states))]

            # Update Q value estimator parameters by optimizing between Q network and Q-learning targets
            loss = agent.train(batch_states, batch_actions, batch_td_target)
        else:
            next_q_values = (target if target else agent).predict([next_obs], None, None)
            best_next_action = np.argmax(next_q_values, axis=1)
            td_target = reward + (gamma * next_q_values[0] * best_next_action)
            loss = agent.train([obs], [[0, action]], td_target)

        epsilon *= epsilon_decay
        obs = next_obs
        # if i_episode % save_interval == 0:
        #     tf_saver.save(sess, save_path, global_step=i_episode)
    return stats

from run_ple_utils import make_ple_env
if __name__ == '__main__':

    # Params
    DISCOUNT = 0.90
    EPSILON = 0.5
    EPS_DECAY = 0.995
    MAX_REPLAY_BUF_SIZE = 10000
    BATCH_SIZE = 50
    NUM_TRAIN_UPDATES = int(8e8)
    TARGET = None
    # save_interval = 500
    LOG_INTERVAL = 30
    DISPLAY_SCREEN = True
    WRITE_SUMMARY = True

    NUMENVS = 1

    seed = 1

    env = make_ple_env('FlappyBird-v1', seed=0)
    q_learning(env, seed=seed,
               total_timesteps=NUM_TRAIN_UPDATES,
               gamma=DISCOUNT,
               epsilon=EPSILON,
               epsilon_decay=EPS_DECAY,
               use_experience_replay=True,
               max_replay_buffer_size=MAX_REPLAY_BUF_SIZE,
               batch_size=BATCH_SIZE,
               target=TARGET,
               # tf_saver=saver,
               # save_path=save_path,
               # save_interval=save_interval,
               log_interval=LOG_INTERVAL,
               # display_screen=True,
               write_summary=WRITE_SUMMARY)
    # learn(CastaPolicy, env, 0, nsteps=50, vf_coef=0.2, ent_coef=1e-7, gamma=0.90, display_screen=True,
    #       lr=5e-5, lrschedule='constant', max_grad_norm=0.01, log_interval=30)
    env.close()


    # game = FlappyBird(pipe_gap=300)
    # env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    #
    # env.init()
    #
    # action_set = env.getActionSet()
    # n_actions = len(action_set)
    # state_dim = env.getGameState().size
    #
    # # Params
    # discount = 0.5
    # epsilon = 0.4
    # epsilon_decay = 0.995
    # max_replay_buffer_size = 1000
    # batch_size = 50
    # num_train_updates = int(8e8)
    # save_interval = 500
    # log_interval = 100
    #
    # nenvs = 1
    # stats = {}
    # with tf.Session() as sess:
    #     agent = DQNAgent(sess, state_dim, n_actions, log_interval=log_interval)
    #     target = None  # TargetNetwork(sess, state_dim, n_actions)
    #
    #     summary_writer = agent.get_summary_writer()
    #     # saver = tf.train.Saver()
    #     # file_name = "DQN_d" + str(discount) + "_e" + str(epsilon) + "_ed" + str(epsilon_decay) + "_batch" + str(batch_size)
    #     # save_path = os.path.join(CHECKPOINT_DIR, file_name)
    #
    #     # sess.run(tf.global_variables_initializer())
    #     # Training
    #     stats[i] = q_learning(sess, env, agent, # num_updates, # max_ep_length,
    #                           total_timesteps=num_train_updates,
    #                           gamma=discount,
    #                           epsilon=epsilon,
    #                           epsilon_decay=epsilon_decay,
    #                           use_experience_replay=True,
    #                           max_replay_buffer_size=max_replay_buffer_size,
    #                           batch_size=batch_size,
    #                           target=target,
    #                           # tf_saver=saver,
    #                           # save_path=save_path,
    #                           # save_interval=save_interval,
    #                           log_interval=log_interval,
    #                           display_screen=True,
    #                           summary_writer=summary_writer)
    # plot_episode_stats(stats)

    # with tf.Session() as sess:
    #     new_saver = tf.train.import_meta_graph(save_path + "-" + str(int(round((num_episodes-1)/100, 0) * 100)) + ".meta")
    #     new_saver.restore(sess, save_path + "-" + str(int(round((num_episodes - 1) / 100, 0) * 100)))
    #     # new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    #
    #     # graph = tf.get_default_graph()
    #     # state_in = graph.get_tensor_by_name("state_in:0")
    #     #
    #     # # for i in tf.get_default_graph().get_operations():
    #     # #     print(i.name)
    #     #
    #     # var_list = tf.get_collection(tf.GraphKeys.TRAIN_OP)
    #     # print(var_list)
    #     #
    #     # action_pred = graph.get_operation_by_name("pred_out")
    #
    #     # Evaluation of trained model
    #     env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    #     for _ in range(100):
    #
    #         for _ in range(1000):
    #             # Update Player parameters gravity and horizontal speeds
    #             #     # if i % 100 == 0:
    #             #         # game.player.GRAVITY *= 0.5 + random.random()
    #             #         # game.player.vel += 40  # game.player.vel sets the y velocity
    #             #         # set the velocity of the moving pipes: add function: set_speed() to Pipe class which adapts
    #             #         # the value of the pipe speed.
    #             if env.game_over():
    #                 env.reset_game()
    #
    #             state = env.getGameState()
    #             # env.act(action_set[np.argmax(sess.run(action_pred, {state_in: state}))])
    #             env.act(action_set[np.argmax(agent.predict(sess, [state]))])
