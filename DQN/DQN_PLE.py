# Deep Q Learning Agent using experience replay and a target network.
import numpy as np
import sys, os, datetime
import tensorflow as tf
import joblib
import logging
# from collections import namedtuple

from models_OAI import DQN, DQN_smac
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

    def __init__(self, ob_space, ac_space, scope, lr, lrschedule, max_grad_norm,
                 units_per_hlayer, batch_size, total_timesteps, log_interval, logdir):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info("Set up DQN learning agent")
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()  # TODO add CPU config information

        nbatch = batch_size
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)


        # Targets in loss computation
        QT = tf.placeholder(shape=[nbatch], dtype=tf.float32)  # target Q values
        A = tf.placeholder(shape=[nbatch, 2], dtype=tf.int32)  # action indices

        # nbatch = self.target_in.shape[1].value if (self.target_in.shape[0].value is not None) else 0

        eval_model = DQN_smac(sess, ob_space, ac_space.n, batch_size=1, units_per_hlayer=units_per_hlayer,
                         scope=scope, reuse=False)
        train_model = DQN_smac(sess, ob_space, ac_space.n, batch_size=batch_size, units_per_hlayer=units_per_hlayer,
                          scope=scope, reuse=True)
        # target_model = TargetNetwork(sess, ob_space, ac_space.n)

        loss = tf.losses.mean_squared_error(labels=QT, predictions=tf.gather_nd(params=train_model.predQ,
                                                                                indices=A))
        params = tf.trainable_variables(scope)
        optimizer = tf.train.AdamOptimizer(lr)
        # self.train_step = self.optimizer.minimize(self.loss)
        gradients = optimizer.compute_gradients(loss)
        grads, variables = zip(*gradients)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        _train = [optimizer.apply_gradients(grads),
                  self.global_step.assign_add(nbatch)]

        if log_interval > 0:
            for g, v in grads:
                if g is not None:
                    tf.summary.histogram("train/grads/%s-grad" % v.name.replace(':', '_'), g)
            for p in params:
                if p is not None:
                    tf.summary.histogram("train/params/%s" % p.name.replace(':', '_'), p.value())
            tf.summary.scalar("train/vf_loss", loss)
            tf.summary.histogram("others/A", A)
            tf.summary.histogram("others/QT", QT)
            self.summary_step = tf.summary.merge_all()

        tf.add_to_collection('inputs', eval_model.X)
        tf.add_to_collection('predQ', eval_model.predQ)
        # tf.add_to_collection('step', eval_model.step)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        tf.global_variables_initializer().run(session=sess)

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
            feed_dict = {train_model.X:obs, A:actions, QT:targets}
            # evaluate the TF tensors and operations self.loss and self.train_step
            total_loss, _, global_step = sess.run([loss, _train, self.global_step], feed_dict=feed_dict)
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                self.logger.info('Save summary of network weights, grads and losses.')
                summary_str = sess.run(self.summary_step, feed_dict)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)
            self.num_steps_trained += 1

            return total_loss

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)
        saver = tf.train.Saver(max_to_keep=7)

        def save(f_name):
            gs = sess.run(self.global_step)
            self.logger.info('Save network parameters of model at global step %s' % gs)
            saver.save(sess, os.path.join(logdir, f_name), global_step=gs)
            # ps = sess.run(params)
            # make_path(save_path)
            # joblib.dump(ps, os.path.join(save_path, f_name))

        def load(load_path):
            saver.restore(sess, load_path)
            # loaded_params = joblib.load(load_path)
            # restores = []
            # for p, loaded_p in zip(params, loaded_params):
            #     restores.append(p.assign(loaded_p))
            # ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = eval_model
        self.step = eval_model.step
        self.predict = eval_model.predict
        # self.target_model = target_model
        self.save = save
        self.load = load

        if log_interval > 0:
            self.summary_writer = tf.summary.FileWriter(logdir, graph_def=sess.graph_def)
        else:
            self.summary_writer = None

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
               max_replay_buffer_size=4000, batch_size=128, lr=5e-4, lrschedule='linear',
               max_grad_norm=0.01, units_per_hlayer=(8,8,8),
               target=None, log_interval= 100, show_interval=0, logdir=None):
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
    logger = logging.getLogger(__name__)
    tf.reset_default_graph()
    set_global_seeds(seed)

    # Params
    ob_space = env.observation_space
    ac_space = env.action_space
    nd, = ob_space.shape
    n_ac = ac_space.n
    use_exp_replay = False if batch_size == 1 else True

    # Create learning agent and the replay buffer
    agent = DQNAgent(ob_space, ac_space, batch_size=batch_size,
                     lr=lr,
                     lrschedule=lrschedule,
                     scope="model",
                     max_grad_norm=max_grad_norm,
                     units_per_hlayer=units_per_hlayer,
                     total_timesteps=total_timesteps,
                     log_interval=log_interval,
                     logdir=logdir)
    summary_writer = agent.get_summary_writer()
    if use_exp_replay:
        replay_buffer = ReplayBuffer(max_replay_buffer_size)

    # Keeps track of useful statistics
    stats = EpisodeStats

    # ------------------ TRAINING --------------------------------------------
    logger.info("Start Training")
    i_episode = 0
    nbatch = batch_size
    max_return = 0

    # Reset env
    obs = env.reset()
    done = False

    for update in range(1, total_timesteps//nbatch + nbatch):  # as we start training only after nbatch experiences

        # Update target network
        if target:
            target.update()

        # action_probs = policy([obs], epsilon)
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        AP = agent.step([obs], epsilon=epsilon)  # epsilon greedy action
        action = np.random.choice(np.arange(n_ac), p=AP)
        next_obs, reward, done, _ = env.step(action)

        # process reward and observations
        reward -= 1e-5
        next_obs = normalize_obs(next_obs)

        # render only every i-th episode
        if show_interval != 0:
            if i_episode % show_interval == 0:
                env.render()

        # episode stats
        stats['episode_lengths'][i_episode] += 1
        stats['episode_rewards'][i_episode] += reward

        # When episode is done, add episode information to tensorboard summary and stats
        if done:  # env.game_over():
            next_obs = np.ones(len(next_obs)) * 0

            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(tag='envs/ep_return',
                                  simple_value=stats['episode_rewards'][i_episode])
                summary.value.add(tag="envs/ep_length",
                                  simple_value=stats['episode_lengths'][i_episode])
                summary_writer.add_summary(summary, i_episode)
                summary_writer.flush()

            # avg_ep_return = np.mean(stats["episode_rewards"][-30:])
            # if avg_ep_return > -4.99 and avg_ep_return >= (max_avg_ep_return + 5):
            #     max_avg_ep_return = avg_ep_return
            #     model.save(logdir, 'inter_model', )

            if stats["episode_rewards"][i_episode] > max_return:
                max_return = stats["episode_rewards"][i_episode]
                logger.info('Save model at max reward %s' % max_return)
                agent.save('inter_model')

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

        # Compute TD target and update the model
        if use_exp_replay:
            # Update replay buffer
            replay_buffer.add_transition(obs, action, next_obs, reward, done)

            if replay_buffer.size() > nbatch:
                # Sample minibatch from replay buffer
                mb_obs, mb_actions, mb_next_obs, mb_rewards, _, batch_dones = \
                    replay_buffer.next_batch(nbatch)

                mb_actions = list(zip(range(len(mb_actions)), mb_actions))

                # Calculate TD target for batch. Use "old" fixed parameters if target network is available
                # to compute targets else use "old" parameters of value function estimate.
                # mb_next_obs = np.reshape(mb_next_obs, (-1, nd))
                mb_next_q_values = (target if target else agent.train_model).predict(mb_next_obs)
                mb_best_next_action = np.argmax(mb_next_q_values, axis=1)
                mb_td_target = [mb_rewards[j] + gamma * mb_next_q_values[j][mb_best_next_action[j]]
                                for j in range(nbatch)]

                # Update Q value estimator parameters by optimizing between Q network and Q-learning targets
                loss = agent.train(mb_obs, mb_actions, mb_td_target)
        else:
            next_q_values = (target if target else agent).predict([next_obs])
            best_next_action = np.argmax(next_q_values, axis=1)
            td_target = reward + (gamma * next_q_values[0] * best_next_action)
            loss = agent.train([obs], [[0, action]], td_target)

        if done:
            # Reset the model
            next_obs = env.reset()
            next_obs = normalize_obs(next_obs)

        epsilon *= epsilon_decay
        obs = next_obs

    # Save final model when training is finished.
    agent.save('final_model')
    logger.info('Finished Training. Saving Final model.\n')

    # return stats

from run_ple_utils import make_ple_env
if __name__ == '__main__':

    # Params
    DISCOUNT = 0.90
    EPSILON = 0.5
    EPS_DECAY = 0.995
    LR=5e-4
    MAX_REPLAY_BUF_SIZE = 10000
    BATCH_SIZE = 50
    MAX_GRAD_NORM=0.5
    NUM_TRAIN_UPDATES = int(2e6)
    TARGET = None
    SAVE_INTERVAL = 500
    LOG_INTERVAL = 30
    DATE = str(datetime.datetime.today())
    LOGDIR = os.path.join('/home/mara/Desktop/logs/DQN', DATE)

    seed = 1
    env = make_ple_env('FlappyBird-v1', seed=seed)
    q_learning(env, seed=seed,
               total_timesteps=NUM_TRAIN_UPDATES,
               gamma=DISCOUNT,
               epsilon=EPSILON,
               epsilon_decay=EPS_DECAY,
               lr=LR,
               max_replay_buffer_size=MAX_REPLAY_BUF_SIZE,
               batch_size=BATCH_SIZE,
               max_grad_norm=MAX_GRAD_NORM,
               units_per_hlayer=(24,24,24),
               target=TARGET,
               log_interval=LOG_INTERVAL,
               show_interval=20,
               logdir=LOGDIR)
    env.close()
