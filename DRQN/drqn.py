# Almost the same as normal DQN agent only differences are:
# In the replay buffer we store sample complete episodes and draw sequential data from each episode
# Therefore it is required to use the clipped environments
# Add rnn cell to model
# Update the network with the samples from nbatch sequences of N timesteps each.
import numpy as np
import os, datetime
import tensorflow as tf
import logging
import csv
from collections import deque

from models_OAI import DRQN  #, DRQN_smac
from utils_OAI import ExperienceBuffer, update_target_graph, add_to_collection_rnn_state
from utils_OAI import Scheduler, make_session, set_global_seeds, normalize_obs

# TODO: implement saving of checkpoints
# TODO: add more statistics values.
# TODO: optimize hyperparameters
# TODO: Prioritize experience in replay buffer
# TODO check initital state
# TODO add replay start value
# summary doesn't work for network params
# How to add target network?
# How to add duelling network?

EpisodeStats = {"episode_lengths": [],
                "episode_rewards": []}


class DRQNAgent():
    """
    Deep Q Network with a recurrent cell used or Deep Q learning
    """
    # Deep Q Learning Agent using experience replay and a target network.

    def __init__(self, ob_space, ac_space, scope, lr, lrschedule, max_grad_norm,
                 units_per_hlayer,
                 nbatch, trace_length,
                 tau, total_timesteps, update_interval,
                 log_interval, logdir, keep_model, activ_fcn): # TODO add rnn params
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info("Set up DQN_RNN learning agent")
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()  # TODO add CPU config information

        # number of samples used for trainign
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)

        # Targets in loss computation
        QT = tf.placeholder(shape=[nbatch*trace_length], dtype=tf.float32, name='QT')  # target Q values
        A = tf.placeholder(shape=[nbatch*trace_length], dtype=tf.int32, name='A')  # action indices

        # nbatch = self.target_in.shape[1].value if (self.target_in.shape[0].value is not None) else 0

        eval_model = DRQN(sess, ob_space, ac_space.n, nbatch=1, trace_length=1, units_per_hlayer=units_per_hlayer,
                              scope='model', reuse=False, activ_fcn=activ_fcn)
        train_model = DRQN(sess, ob_space, ac_space.n, nbatch=nbatch, trace_length=trace_length, units_per_hlayer=units_per_hlayer,
                               scope='model', reuse=True, activ_fcn=activ_fcn)
        target_model = DRQN(sess, ob_space, ac_space.n, nbatch=nbatch, trace_length=trace_length, units_per_hlayer=units_per_hlayer,
                             scope='target', reuse=False, activ_fcn=activ_fcn)

        # Obtain loss by taking the mean of squares difference between the target and prediction Q values.
        actions_onehot = tf.one_hot(A, depth=ac_space.n, dtype=tf.float32)
        td_error = tf.losses.mean_squared_error(labels=QT, predictions=tf.squeeze(tf.matmul(tf.multiply(train_model.predQ, actions_onehot), [[1.],[1.]])))
        # td_error = tf.losses.mean_squared_error(labels=QT, predictions=tf.gather_nd(params=train_model.predQ,
        #                                                                             indices=A))
        loss = td_error # * mask   # where mask is a mask that masks the first half of the losses to 0 for each trace
                                   # as the rnn state does not have enough information on their history yet in order
                                   # to only propogate accurate gradients through the network as per
                                   # Lample & Chatlot 2016 "Plaxing FPS Games iwth DRL":
        # "A sequence of n observations o_1,o_2, ..., o_n is randomly sampled from the replay memory, but instead of
        # updating all action states in the sequence, we only consider the ones that are provided with enough history.
        # Indeed, the first states of the sequence will be from an almost non-existent history(since h_0 is
        # reinitialized at the beginning of thte updates), and might be inaccurate. As a result, updatinf them might
        # lead to imprecise results."

        params = tf.trainable_variables()  # TODO which trainable variables do I want here? 'model', 'target' or all??  Rather all as I need model and target variables to update the target model
        optimizer = tf.train.AdamOptimizer(lr)
        # self.train_step = self.optimizer.minimize(self.loss)
        gradients = optimizer.compute_gradients(loss)
        grads, variables = zip(*gradients)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        _train = [optimizer.apply_gradients(grads),
                  self.global_step.assign_add(update_interval)]  # nbatch  # TODO initally add the number of steps done before training for the first time

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

        tf.add_to_collection('inputs', eval_model.obs_in)
        add_to_collection_rnn_state('state_in', eval_model.rnn_state_in)
        add_to_collection_rnn_state('state_out', eval_model.rnn_state_out)
        tf.add_to_collection('predQ', eval_model.predQ)
        # tf.add_to_collection('step', eval_model.step)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        tf.global_variables_initializer().run(session=sess)

        def train(obs, actions, targets, rnn_states):
            """
            Updates the weights of the neural network, based on its targets, its
            predictions, its loss and its optimizer.

            Args:
                sess: TensorFlow session.
                obs: [current_observation] or observations of batch
                actions: [current_action] or actions of batch
                targets: [current_target] or targets of batch
            """
            feed_dict = {train_model.obs_in: obs, A: actions, QT: targets, train_model.rnn_state_in: rnn_states}
            # evaluate the TF tensors and operations self.loss and self.train_step
            total_loss, _, global_step = sess.run([loss, _train, self.global_step], feed_dict=feed_dict)
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                self.logger.info('Save summary of network weights, grads and losses.')
                summary_str = sess.run(self.summary_step, feed_dict)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), self.num_steps_trained)  # global_step)

            # In config evaluation mode save model regularly to track performance of the learning agent.
            # if save_interval > 0 and (self.num_steps_trained % save_interval == 0):
            #     self.logger.info('Save model at train step %s' % self.num_steps_trained)
            #     self.save('config_model')

            self.num_steps_trained += 1

            return total_loss

        # tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.pred_out)
        saver = tf.train.Saver(max_to_keep=7)
        # saver = tf.train.Saver(max_to_keep=keep_model)

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
            self.logger.debug('Evaluating eval_model')
            ep_return = []
            ep_length = []
            for i in range(0, n_eps):  # TODO parallelize this here!
                obs = env.reset()
                obs = normalize_obs(obs)
                done = False
                rnn_s_in = (np.zeros(eval_model.rnn_state_in[0].shape), np.zeros(eval_model.rnn_state_in[1].shape))  # init rnn cell vector
                total_return = 0
                total_length = 0

                while not done and (total_return < n_pipes):
                    pQ, rnn_s_out = sess.run([eval_model.predQ, eval_model.rnn_state_out], feed_dict={eval_model.obs_in: [obs], eval_model.rnn_state_in: rnn_s_in})
                    ac = np.argmax(pQ)
                    obs, reward, done, _ = env.step(ac)
                    # obs, reward, done, _ = env.step(act[0][0])
                    obs = normalize_obs(obs)
                    total_length += 1
                    total_return += reward
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
        self.predict = target_model.predict
        self.save = save
        self.load = load
        self.test_run = test_run

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
#         self.sess = sess
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
#                                                                    + ((1 - self.tau) * tf_vars[
#                 idx + total_vars // 2].value())))
#         return op_holder

def q_learning(env, test_env, seed, total_timesteps=int(1e8), gamma=0.95, epsilon=0.4, epsilon_decay=.95, tau=0.99,
               buffer_size=4000, nbatch=128, trace_length=32, lr=5e-4, lrschedule='linear',
               max_grad_norm=0.01, units_per_hlayer=(8, 8, 8), # pre_train_steps=1000,
               scope='model', update_interval=5, log_interval=100, test_interval=0,
               show_interval=0, logdir=None, keep_model=7, activ_fcn='relu6'):
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
    logger.info(datetime.time())
    tf.reset_default_graph()
    set_global_seeds(seed)

    # Params
    ob_space = env.observation_space
    ac_space = env.action_space
    nd, = ob_space.shape
    n_ac = ac_space.n
    # use_exp_replay = False if nbatch == 1 else True

    # Create learning agent and the replay buffer
    agent = DRQNAgent(ob_space=ob_space,
                      ac_space=ac_space,
                      scope=scope,
                      lr=lr,
                      lrschedule=lrschedule,
                      max_grad_norm=max_grad_norm,
                      units_per_hlayer=units_per_hlayer,
                      nbatch=nbatch,
                      trace_length=trace_length,
                      tau=tau,
                      total_timesteps=total_timesteps,
                      update_interval=update_interval,
                      log_interval=log_interval,
                      logdir=logdir,
                      keep_model=keep_model,
                      activ_fcn=activ_fcn)
    summary_writer = agent.get_summary_writer()
    result_path = os.path.join(logdir, 'results.csv')

    sample_size = 5  # [s, a, r, s1, d]
    replay_buffer = ExperienceBuffer(buffer_size, sample_size=sample_size)

    # Keeps track of useful statistics
    stats = EpisodeStats

    # ------------------ TRAINING --------------------------------------------
    logger.info("Start Training")
    i_episode, i_sample, i_train = 0, 0, 0
    len, rew = 0, 0
    rolling_mean = deque(maxlen=30)
    return_treshold = 40  # models which achieve higher total return get stored.

    # Reset env
    obs = env.reset()
    obs = normalize_obs(obs)
    done = False

    # The rnn state consists of the "cell state" c and the "input vector" x_t = h_{t-1}
    rnn_state0 = (np.zeros([1, units_per_hlayer[2]]), np.zeros([1, units_per_hlayer[2]]))   #  units_per_hlayer[2] x  units_per_hlayer[2] matrix

    episode_buffer = []

    # Set the target network to be equal to the primary network
    agent.update_target(agent.target_ops)
    # for update in range(1, total_timesteps // nbatch + nbatch):  # as we start training only after nbatch experiences
    while i_sample < total_timesteps:
        # # Update target network
        # if target:
        #     target.update()

        # Choose epsilon-greedy action (with epsilon being the chance of random action) using the network
        if np.random.rand(1) < epsilon:
            _, next_rnn_state = agent.step_model.step([obs], rnn_state0)
            action = np.random.randint(0, n_ac)
        else:
            # print(obs)  # TODO do I get the right action here?
            # print(agent.step_model.predict([obs], rnn_state0))
            AP, next_rnn_state = agent.step_model.step([obs], rnn_state0)
            action = AP[0]
            # print(action)
            # action = np.random.choice(np.arange(n_ac), p=AP)

        # AP = agent.step([obs], epsilon=epsilon)  # epsilon greedy action
        # action = np.random.choice(np.arange(n_ac), p=AP)
        next_obs, reward, done, _ = env.step(action)  #action)
        # print('rew %s' % reward)
        next_obs = normalize_obs(next_obs)
        reward -= 1e-5
        i_sample += 1

        # render only every i-th episode
        if show_interval != 0:
            if i_episode % show_interval == 0:
                env.render()

        # episode stats
        # stats['episode_lengths'][i_episode] += 1
        # stats['episode_rewards'][i_episode] += reward
        len += 1  # TODO check whether this works
        rew += reward
        rolling_mean.append(reward)

        # When episode is done, add episode information to tensorboard summary and stats
        if done:  # env.game_over():
            next_obs = list(np.zeros_like(next_obs, dtype=np.float64))
            episode_buffer.append(np.reshape(np.array([obs, action, reward, next_obs, done]), newshape=[1, 5]))

            replay_buffer.add(list(zip(np.array(episode_buffer))))  # TODO does thos here lead to object dtype?
            episode_buffer = []
            stats['episode_lengths'].append(len)
            stats['episode_rewards'].append(rew)

            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(tag='envs/ep_return',
                                  simple_value=stats['episode_rewards'][i_episode])
                summary.value.add(tag="envs/ep_length",
                                  simple_value=stats['episode_lengths'][i_episode])
                summary_writer.add_summary(summary, i_episode)
                summary_samples = tf.Summary()
                summary_samples.value.add(tag='envs/samples/ep_return',
                                  simple_value=stats['episode_rewards'][i_episode])
                summary_samples.value.add(tag="envs/samples/ep_length",
                                  simple_value=stats['episode_lengths'][i_episode])
                summary_writer.add_summary(summary_samples, i_sample)
                summary_writer.flush()

            if rew > return_treshold:
                return_treshold = rew
                logger.info('Save model at max reward %s' % return_treshold)
                agent.save('inter_model')
            i_episode += 1
            # print(i_episode)
            len, rew = 0, 0
            # stats['episode_lengths'].append(0)
            # stats['episode_rewards'].append(0)
        else:
            episode_buffer.append(np.reshape(np.array([obs, action, reward, next_obs, done]), newshape=[1, 5]))

        # Compute TD target and update the model from the sampled traces in the buffer as soon as #pre_train_steps
        # where done with the environments
        if i_episode >= nbatch:  # if number of finished episodes is greather or equal the number of episodes
                                 # from which traces are sampled.
            if i_sample % update_interval == 0:
                # TODO update epsilon

                # update target
                # start_time = datetime.time()
                agent.update_target(agent.target_ops)
                # logger.info('Update Target duration: %s - %s' % (start_time, datetime.time()))

                # reset rnn state (history knowledge) before every training step
                rnn_state_train = (np.zeros([nbatch, units_per_hlayer[2]]), np.zeros([nbatch, units_per_hlayer[2]]))

                # sample training batch from replay buffer
                training_batch = replay_buffer.sample(nbatch=nbatch, trace_length=trace_length)

                mb_obs = training_batch[:,0].tolist()
                mb_actions = training_batch[:,1].astype(np.int32)
                mb_rewards = training_batch[:,2].astype(np.float64)
                mb_next_obs = training_batch[:,3].tolist()
                mb_dones = training_batch[:,4].astype(bool)

                # Compute target Q values for the given batch
                mb_next_q_values, _ = agent.target_model.predict(mb_next_obs, rnn_state=rnn_state_train)
                mb_best_next_action = np.argmax(mb_next_q_values, axis=1)
                mb_td_target = [mb_rewards[j] + gamma * mb_next_q_values[j][mb_best_next_action[j]]
                                for j in range(nbatch*trace_length)]

                # train model
                # start_time = datetime.time()
                loss = agent.train(mb_obs, mb_actions, mb_td_target, rnn_state_train)
                # logger.info('Train duration: %s - %s' % (start_time, datetime.time()))
                i_train += 1

                # If test_interval > 0 the learned model is evaluated every "test_interval" gradient updates
                if test_interval > 0 and i_train > 0 and (i_train % test_interval == 0):
                    # print('testing')
                    ep_return = agent.test_run(test_env, n_eps=30, n_pipes=2000)
                    with open(result_path, "a") as csvfile:
                        writer = csv.writer(csvfile)
                        ep_return = [str(p) for p in ep_return]
                        ep_return.insert(0, ('step_%s_eps_%s' % (i_sample, i_episode)))
                        writer.writerow(ep_return)
        if done:
            # Reset the model
            next_obs = env.reset()
            next_obs = normalize_obs(next_obs)

        epsilon *= epsilon_decay
        obs = next_obs
        rnn_state0 = next_rnn_state

    # Save final model when training is finished.
    agent.save('final_model')
    logger.info('Finished Training. Saving Final model.')

    logger.info('*******************************************************')
    logger.info('Total number of interactions with the environment: %s' % i_sample)
    logger.info('Total number of finished episodes during training: %s' % i_episode)
    logger.info('Total number of parameter updates during training: %s' % i_train)
    logger.info('*******************************************************\n')
    # return stats

from run_ple_utils import make_ple_env
if __name__ == '__main__':
    # Params
    DISCOUNT = 0.90
    EPSILON = 0.5
    EPS_DECAY = 0.995
    LR = 5e-4
    MAX_REPLAY_BUF_SIZE = 1000
    BATCH_SIZE = 4  # number of episodes from which trces are sampled
    MAX_GRAD_NORM = 0.5
    NUM_TRAIN_UPDATES = int(2e6)
    TARGET = None
    SAVE_INTERVAL = 500
    LOG_INTERVAL = 30
    DATE = str(datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    LOGDIR = os.path.join('/home/mara/Desktop/logs/DQN', DATE)

    seed = 2
    env = make_ple_env('FlappyBird-v1', seed=seed)
    test_env = make_ple_env('FlappyBird-v1', seed=seed)

    q_learning(env, test_env=test_env, seed=seed,
               total_timesteps=NUM_TRAIN_UPDATES,
               gamma=DISCOUNT,
               epsilon=EPSILON,
               epsilon_decay=EPS_DECAY,
               tau=0.90,
               lr=LR,
               buffer_size=MAX_REPLAY_BUF_SIZE,
               nbatch=BATCH_SIZE,
               trace_length=8,
               max_grad_norm=MAX_GRAD_NORM,
               units_per_hlayer=(24, 24, 24),
               # pre_train_steps=1000,
               update_interval=5,
               log_interval=LOG_INTERVAL,
               test_interval=10,
               show_interval=0,
               logdir=LOGDIR,
               keep_model=7,
               activ_fcn='relu6')
    env.close()
