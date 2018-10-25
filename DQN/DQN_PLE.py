# Deep Q Learning Agent using experience replay and a target network.
import numpy as np
import sys, os, datetime, time
import csv
import tensorflow as tf
# import joblib
import logging
from collections import deque
# from collections import namedtuple

from utils_OAI import ReplayBuffer
from utils_OAI import normalize_obs, update_target_graph, add_to_collection_rnn_state
from utils_OAI import make_session, set_global_seeds

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

    def __init__(self, q_network, ob_space, ac_space,
                 lr, max_grad_norm, units_per_hlayer, activ_fcn, log_interval, logdir,
                 batch_size, trace_length, tau, update_interval, keep_model):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info("Set up DQN learning agent")
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()  # TODO add CPU config information

        # nbatch = batch_size
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)


        # Targets in loss computation
        QT = tf.placeholder(shape=[batch_size*trace_length], dtype=tf.float32, name='QT')  # target Q values
        A = tf.placeholder(shape=[batch_size*trace_length], dtype=tf.int32, name='A')  # action indices

        # nbatch = self.target_in.shape[1].value if (self.target_in.shape[0].value is not None) else 0

        eval_model = q_network(sess, ob_space, ac_space.n, nbatch=1, trace_length=1, units_per_hlayer=units_per_hlayer,
                         scope='model', reuse=False, activ_fcn=activ_fcn)
        train_model = q_network(sess, ob_space, ac_space.n, nbatch=batch_size, trace_length=trace_length, units_per_hlayer=units_per_hlayer,
                          scope='model', reuse=True, activ_fcn=activ_fcn)
        # target_model = TargetNetwork(sess, ob_space, ac_space.n)
        target_model = q_network(sess, ob_space, ac_space.n, nbatch=batch_size, trace_length=trace_length, units_per_hlayer=units_per_hlayer,
                               scope='target', reuse=False, activ_fcn=activ_fcn)

        # Obtain loss by taking the mean of squares difference between the target and prediction Q values.
        actions_onehot = tf.one_hot(A, depth=ac_space.n, dtype=tf.float32)
        td_error = tf.losses.mean_squared_error(labels=QT,
                                                predictions=tf.squeeze(tf.matmul(tf.multiply(train_model.predQ, actions_onehot), [[1.], [1.]])))
        loss = td_error
        # loss = tf.losses.mean_squared_error(labels=QT, predictions=tf.gather_nd(params=train_model.predQ,
        #                                                                         indices=A))
        params = tf.trainable_variables()  # was set to 'model', but we would need model and target parameters
        optimizer = tf.train.AdamOptimizer(lr)
        # self.train_step = self.optimizer.minimize(self.loss)
        gradients = optimizer.compute_gradients(loss)
        grads, variables = zip(*gradients)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        _train = [optimizer.apply_gradients(grads),
                  self.global_step.assign_add(update_interval)]  # nbatch

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
        if eval_model.initial_state is not None:
            add_to_collection_rnn_state('state_in', eval_model.rnn_state_in)
            add_to_collection_rnn_state('state_out', eval_model.rnn_state_out)
        # tf.add_to_collection('step', eval_model.step)

        tf.global_variables_initializer().run(session=sess)

        def train(obs, actions, targets, states):
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
            if states is not None:
                feed_dict[train_model.rnn_state_in] = states
            # evaluate the TF tensors and operations self.loss and self.train_step
            total_loss, _, global_step = sess.run([loss, _train, self.global_step], feed_dict=feed_dict)
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                self.logger.info('Save summary of network weights, grads and losses.')
                summary_str = sess.run(self.summary_step, feed_dict)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)

            # # In config evaluation mode save model regularly to track performance of the learning agent.
            # if save_interval > 0 and (self.num_steps_trained % save_interval == 0):
            #     self.logger.info('Save model at train step %s' % self.num_steps_trained)
            #     self.save('config_model')

            self.num_steps_trained += 1

            return total_loss

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)
        # saver = tf.train.Saver(max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=keep_model)

        def update_target(target_op_holder):
            for op in target_op_holder:
                sess.run(op)
            a = tf.trainable_variables()[0].eval(session=sess)
            b = tf.trainable_variables()[len(params)//2].eval(session=sess)
            if not a.all() == b.all():
                print("Target Set Failed")
                # print("Target Set Success")
            # else:
            #     print("Target Set Failed")

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

        def test_run(env, n_eps, n_pipes):
            self.logger.info('Evaluating current agent')
            ep_return = []
            ep_length = []
            for i in range(0, n_eps):  # TODO parallelize this here!
                obs = env.reset()
                obs = normalize_obs(obs)
                done = False
                if eval_model.initial_state is not None:
                    if len(eval_model.initial_state) > 1:
                        rnn_s_in = (np.zeros(eval_model.initial_state[0].shape), np.zeros(eval_model.initial_state[1].shape))  # init lstm cell vector
                    else:
                        rnn_s_in = np.zeros(eval_model.initial_state.shape)  # init gru cell vector
                    # rnn_s_in = (np.zeros([1, units_per_hlayer[2]]), np.zeros([1, units_per_hlayer[2]]))
                total_return = 0
                total_length = 0

                while not done and (total_return < n_pipes):
                    if eval_model.initial_state is not None:
                        pQ, rnn_s_out = sess.run([eval_model.predQ, eval_model.rnn_state_out],
                                                 feed_dict={eval_model.X: [obs], eval_model.rnn_state_in: rnn_s_in})
                    else:
                        pQ = sess.run([eval_model.predQ], feed_dict={eval_model.X: [obs]})
                    ac = np.argmax(pQ)
                    obs, reward, done, _ = env.step(ac)
                    # obs, reward, done, _ = env.step(act[0][0])
                    obs = normalize_obs(obs)
                    total_length += 1
                    total_return += reward
                    # print(reward)
                    if eval_model.initial_state is not None:
                        rnn_s_in = rnn_s_out
                # logger.debug('*****************************************')
                self.logger.info('Episode %s: %s, %s' % (i, total_return, total_length))
                ep_length.append(total_length)
                ep_return.append(total_return)
            return ep_return

        self.train = train
        self.train_model = train_model
        self.step_model = eval_model
        self.target_model = target_model
        self.target_ops = update_target_graph(params, tau)  # TODO implement update_target_graph

        self.update_target = update_target

        self.step = eval_model.step
        self.predict = eval_model.predict
        self.step_initial_state = eval_model.initial_state
        self.train_initial_state = train_model.initial_state
        self.save = save
        self.load = load
        self.test_run = test_run
        self.sess = sess

        if log_interval > 0:
            self.summary_writer = tf.summary.FileWriter(logdir, graph_def=sess.graph_def)
        else:
            self.summary_writer = None

    def get_summary_writer(self):
        return self.summary_writer


# class TargetNetwork(DQN):
#     """
#     Slowly updated target network. Tau indicates the speed of adjustment. If 1,
#     it is always set to the values of its associate.
#     """
#
#     def __init__(self, sess, state_dimension, num_actions, tau=0.001):
#         self. sess = sess
#         DQN.__init__(self, sess, state_dimension, num_actions, scope="model", reuse=True)
#         self.tau = tau
#         self._counterpart = self._register_counterpart()
#
#         def update():
#             for op in self._counterpart:
#                 self.sess.run(op)
#
#         self.update = update
#
#     def _register_counterpart(self):
#         tf_vars = tf.trainable_variables()
#         total_vars = len(tf_vars)
#         # (total_vars)
#         op_holder = []
#         for idx, var in enumerate(tf_vars[0:total_vars // 2]):
#             # taken from: https://arxiv.org/pdf/1509.02971.pdf
#             # theta' = tau * theta + (1 - tau) * theta'
#             # where theta' is the parameter of the target network.
#             print(var)
#             print(var.value())
#             op_holder.append(tf_vars[idx + total_vars // 2].assign((var.value() * self.tau)
#                                         + ((1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
#         return op_holder


def q_learning(q_network, env, test_env, seed, total_timesteps,
               log_interval, test_interval, show_interval, logdir,
               lr, max_grad_norm, units_per_hlayer, activ_fcn,
               gamma=0.95, epsilon=0.4, epsilon_decay=.95,
               buffer_size=4000, batch_size=128, trace_length=32,
               tau=0.99, update_interval=30, early_stop=False, keep_model=2,
               save_model=True, restore_model=False, save_traj=False):
               #  lr=5e-4,
               # max_grad_norm=0.01, units_per_hlayer=(8,8,8),
               # target=None, log_interval= 100, test_interval=0, show_interval=0, update_interval=30, logdir=None,
               # keep_model=7, activ_fcn='relu6', tau=0.99):
    # """
    # Q-Learning algorithm for off-policy TD control using Function Approximation.
    # Finds the optimal greedy policy while following an epsilon-greedy policy.
    # Implements the options of online learning or using experience replay and also
    # target calculation by target networks, depending on the flags. You can reuse
    # your Q-learning implementation of the last exercise.
    #
    # Args:
    #     env: PLE game
    #     approx: Action-Value function estimator
    #     num_episodes: Number of episodes to run for.
    #     max_time_per_episode: maximum number of time steps before episode is terminated
    #     discount_factor: gamma, discount factor of future rewards.
    #     epsilon: Chance to sample a random action. Float betwen 0 and 1.
    #     epsilon_decay: decay rate of epsilon parameter
    #     use_experience_replay: Indicator if experience replay should be used.
    #     batch_size: Number of samples per batch.
    #     target: Slowly updated target network to calculate the targets. Ignored if None.
    #
    # Returns:
    #     An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    # """
    logger = logging.getLogger(__name__)
    # logger.info(datetime.time)

    tf.reset_default_graph()
    set_global_seeds(seed)

    # Params
    ob_space = env.observation_space
    ac_space = env.action_space
    nd, = ob_space.shape
    n_ac = ac_space.n

    # Create learning agent and the replay buffer
    agent = DQNAgent(q_network=q_network,
                     ob_space=ob_space,
                     ac_space=ac_space,
                     lr=lr,
                     max_grad_norm=max_grad_norm,
                     units_per_hlayer=units_per_hlayer,
                     activ_fcn=activ_fcn,
                     log_interval=log_interval,
                     logdir=logdir,

                     batch_size=batch_size,
                     trace_length=trace_length,
                     update_interval=update_interval,
                     tau=tau,
                     keep_model=keep_model)
    summary_writer = agent.get_summary_writer()
    result_path = os.path.join(logdir, 'train_results.csv')
    if save_traj:
        rew_traj = []
        rew_results_path = os.path.join(logdir, ('lr'+str(lr)+'_tracking_results.csv'))
    else:
        rew_results_path = None
    replay_buffer = ReplayBuffer(buffer_size)

    # Keeps track of useful statistics
    stats = EpisodeStats

    if restore_model:
        for el in os.listdir(logdir):
            if 'final' in el and '.meta' in el:
                # Load pre trained model and set network parameters
                logger.info('load %s' % os.path.join(logdir, el[:-5]))
                agent.load(os.path.join(logdir, el[:-5]))
                # Reset global step parameter.
                agent.sess.run(agent.global_step.assign(0))

    # ------------------ TRAINING --------------------------------------------
    logger.info("Start Training")
    early_stopped = False
    i_episode, i_sample, i_train = 0, 0, 0
    len, rew = 0, 0
    horizon = 100
    reward_window = deque(maxlen=horizon)
    avg_rm = deque(maxlen=30)
    nbatch = batch_size * trace_length
    return_threshold = -0.05  # 40

    # Reset envnn
    obs = env.reset()
    obs = normalize_obs(obs)
    done = False
    rnn_state0 = agent.step_initial_state
    if rnn_state0 is None:  # If we use a normal feed forward architecture, we sample a batch of single samples, not a batch of sequences.
        trace_length = 1

    # Set the target network to be equal to the primary network
    agent.update_target(agent.target_ops)

    # for update in range(1, total_timesteps//nbatch + nbatch):  # as we start training only after nbatch experiences
    while i_sample < total_timesteps:

        # # Update target network
        # if target:
        #     target.update()

        # action_probs = policy([obs], epsilon)
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # AP = agent.step([obs], epsilon=epsilon)  # epsilon greedy action
        # action = np.random.choice(np.arange(n_ac), p=AP)
        if np.random.rand(1) < epsilon:
            _, next_rnn_state = agent.step([obs], rnn_state0)  # epsilon greedy action
            action = np.random.randint(0, n_ac)
        else:
            AP, next_rnn_state = agent.step([obs], rnn_state0)  # epsilon greedy action
            # print(AP, agent.predict([obs]))
            action = AP[0]
        next_obs, reward, done, _ = env.step(action)
        next_obs = normalize_obs(next_obs)
        # print(next_obs)
        # reward -= 1e-5
        # print(reward, next_obs)
        i_sample += 1
        # render only every i-th episode
        if show_interval != 0:
            if i_episode % show_interval == 0:
                env.render()
                # time.sleep(1)

        # episode stats
        # stats['episode_lengths'][i_episode] += 1
        # stats['episode_rewards'][i_episode] += reward
        len += 1
        rew += reward
        reward_window.append(reward)

        # When episode is done, add episode information to tensorboard summary and stats
        if done:  # env.game_over():
            next_obs = list(np.zeros_like(next_obs, dtype=np.float32))

            stats['episode_lengths'].append(len)
            stats['episode_rewards'].append(rew)

            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(tag='envs/ep_return',
                                  simple_value=stats['episode_rewards'][i_episode])
                summary.value.add(tag="envs/ep_length",
                                  simple_value=stats['episode_lengths'][i_episode])
                summary_writer.add_summary(summary, i_episode)
                summary_writer.flush()

            if save_model and rew > return_threshold:
                return_threshold = rew
                logger.info('Save model at max reward %s' % return_threshold)
                agent.save('inter_model')

            i_episode += 1
            len, rew = 0, 0

        # Update replay buffer
        replay_buffer.add_transition(obs, action, next_obs, reward, done)
        rew_traj.append(reward)

        # Update model parameters every #update_interval steps. Use real experience and replayed experience.
        if replay_buffer.size() > nbatch and (i_sample % update_interval == 0):
            if (env.spec._env_name == 'ContFlappyBird'):
                rm = sum(reward_window) / horizon
                if summary_writer is not None:
                    s_summary = tf.Summary()
                    s_summary.value.add(tag='envs/isample_return',
                                      simple_value=rm)
                    summary_writer.add_summary(s_summary, i_sample)

                    # t_summary = tf.Summary()
                    # t_summary.value.add(tag='envs/itrain_return',
                    #                   simple_value=rm)
                    # summary_writer.add_summary(t_summary, i_train)
                    summary_writer.flush()

                # avg_ep_return = np.mean(stats["episode_rewards"][-30:])
                # if avg_ep_return > -4.99 and avg_ep_return >= (max_avg_ep_return + 5):
                #     max_avg_ep_return = avg_ep_return
                #     model.save(logdir, 'inter_model', )

                if save_model and rm > return_threshold:
                    return_threshold = rm
                    logger.info('Save model at max rolling mean %s' % return_threshold)
                    agent.save('inter_model')

                avg_rm.append(rm)

            if early_stop:
                if (i_sample > 60000) and (i_sample <= (60000 + update_interval)): # TODO how to determine early-stopping criteria non-heuristically, but automatically? - BOHB algorithm?
                    if (sum(avg_rm) / 30) <= -0.88:
                        print('breaked')
                        early_stopped = True
                        break

            # TODO implement performance measure which optimizes time until convergence. Convergence defined as avg of last 30 moving averages with horizon 100??
            # if (sum(avg_rm)/30) > 0.0:
            #     break  # and return i_sample but only do this if performance measure =

            agent.update_target(agent.target_ops)

            # reset rnn state (history knowledge) before every training step
            rnn_state_train = agent.train_initial_state

            # Sample training mini-batch from replay buffer
            if rnn_state_train is not None:
                mb_obs, mb_actions, mb_next_obs, mb_rewards, _, batch_dones = \
                                                replay_buffer.recent_and_next_batch_of_seq(batch_size, trace_length)
            else:
                mb_obs, mb_actions, mb_next_obs, mb_rewards, _, batch_dones = \
                                                replay_buffer.recent_and_next_batch(batch_size)
            # mb_obs, mb_actions, mb_next_obs, mb_rewards, _, batch_dones = \
            #     replay_buffer.next_batch(nbatch)
            # mb_actions = list(zip(range(mb_actions.shape[0]), mb_actions))  # TODO check whether labeling of each sequence works.
            # mb_actions = list(zip(range(mb_actions.__len__()), mb_actions))  # TODO check whether labeling of each sequence works.

            # Calculate TD target for batch. Use "old" fixed parameters if target network is available
            # to compute targets else use "old" parameters of value function estimate.
            # mb_next_obs = np.reshape(mb_next_obs, (-1, nd))
            mb_next_q_values, _ = agent.target_model.predict(mb_next_obs, rnn_state_train)
            mb_best_next_action = np.argmax(mb_next_q_values, axis=1)
            mb_td_target = [mb_rewards[j] + gamma * mb_next_q_values[j][mb_best_next_action[j]]
                            for j in range(nbatch)]

            # Update Q value estimator parameters by optimizing between Q network and Q-learning targets
            loss = agent.train(mb_obs, mb_actions, mb_td_target, rnn_state_train)
            logger.info('Trained')
            i_train += 1

            # If test_interval > 0 the learned model is evaluated every "test_interval" gradient updates
            if test_interval > 0 and i_train > 0 and (i_train % test_interval == 0):
                # print('testing')
                ep_return = agent.test_run(test_env, n_eps=10, n_pipes=2000)
                # print(ep_return)
                with open(result_path, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    # ep_return = [str(p) for p in ep_return]
                    # ep_return.insert(0, ('step_%s' % i_sample))
                    ep_return[0:0] = [i_sample, i_train]  # TODO test wehther results has appropriate format of i_sample, i_train, eps1, eps2, eps3, ... here
                    writer.writerow(ep_return)
        # else:
        #     next_q_values = (target if target else agent).predict([next_obs])
        #     best_next_action = np.argmax(next_q_values, axis=1)
        #     td_target = reward + (gamma * next_q_values[0] * best_next_action)
        #     loss = agent.train([obs], [[0, action]], td_target, None)
        #     i_train += 1
        #
        #     # If test_interval > 0 the learned model is evaluated every "test_interval" gradient updates
        #     if test_interval > 0 and i_train > 0 and (i_train % test_interval == 0):
        #         ep_return = agent.test_run(test_env, n_eps=1, n_pipes=2000)
        #         with open(result_path, "a") as csvfile:
        #             writer = csv.writer(csvfile)
        #             ep_return = [str(p) for p in ep_return]
        #             ep_return.insert(0, ('step_%s' % i_sample))
        #             writer.writerow(ep_return)

        if done:
            # Reset the model
            next_obs = env.reset()
            next_obs = normalize_obs(next_obs)

        epsilon *= epsilon_decay
        obs = next_obs
        rnn_state0 = next_rnn_state

    # Save final model when training is finished.
    if save_model:
        agent.save('final_model')
        logger.info('Finished Training. Saving Final model.')

    if rew_results_path is not None:
        logger.info('Save reward trajectory to %s' % rew_results_path)
        with open(rew_results_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            traj = np.asanyarray(rew_traj).reshape(-1).tolist()
            traj[0:0] = [np.mean(traj)]  # i_train, i_sample
            writer.writerow(traj)

    logger.info('*******************************************************')
    logger.info('Total number of interactions with the environment: %s' % i_sample)
    # logger.info('Total number of finished episodes during training: %s' % i_episode)
    logger.info('Total number of parameter updates during training: %s' % i_train)
    # logger.info('Training was early stopped: %s' % early_stopped)
    logger.info('*******************************************************\n')

    return early_stopped, i_sample


if __name__ == '__main__':
    from models_OAI import DQN, DQN_smac, LSTM_DQN, GRU_DQN
    from run_ple_utils import make_ple_env

    # TODO batch size has to be smaller than
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Params
    # BOHB config LOW / SMAC config 3
    ACTIV_FCN = 'elu'
    DISCOUNT = 0.82
    EPSILON = 0.35
    EPS_DECAY = 0.990
    TAU = 0.824
    LR = 0.006
    LAYER1 = 71
    LAYER2 = 65
    LAYER3 = 85
    BATCH_SIZE = 64

    MAX_REPLAY_BUF_SIZE = 500
    BATCH_SIZE = 64
    MAX_GRAD_NORM=0.1
    NUM_TRAIN_UPDATES = int(2e3)
    # TARGET = True
    # SAVE_INTERVAL = 500
    LOG_INTERVAL = 30
    # DATE = datetime.datetime.today().strftime('%Y_%m_%d_%H%M%S')  #str(datetime.datetime.today())
    LOGDIR = os.path.join('/home/mara/Desktop/logs/LSTM_DQN_test', 'dqn_output2')

    seed = 1
    env = make_ple_env('ContFlappyBird-hNS-nrf0-train-v0', seed=seed)
    test_env = make_ple_env('ContFlappyBird-v3', seed=seed)
    # env = make_ple_env('FlappyBird-v1', seed=seed)
    # test_env = make_ple_env('FlappyBird-v1', seed=seed)
    q_learning(q_network=LSTM_DQN,
               env=env,
               test_env=test_env,
               seed=seed,
               total_timesteps=NUM_TRAIN_UPDATES,
               log_interval=LOG_INTERVAL,
               test_interval=0,
               show_interval=0,
               logdir=LOGDIR,
               # keep_model=7,
               lr=LR,
               max_grad_norm=MAX_GRAD_NORM,
               units_per_hlayer=(LAYER1,LAYER2,LAYER3),
               activ_fcn=ACTIV_FCN,
               gamma=DISCOUNT,
               epsilon=EPSILON,
               epsilon_decay=EPS_DECAY,
               buffer_size=MAX_REPLAY_BUF_SIZE,
               batch_size=5,
               trace_length=8,
               # target=TARGET,
               tau=TAU,
               update_interval=64,
               save_traj=True)
    env.close()
