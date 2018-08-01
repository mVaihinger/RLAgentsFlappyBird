import os, sys
import gym
import time
import datetime
# import joblib
import csv
import logging
import numpy as np
import tensorflow as tf
# import logger
from collections import deque
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session, set_global_seeds
from utils_OAI import cat_entropy, mse, explained_variance, normalize_obs, discount_with_dones
from utils_OAI import add_to_collection_rnn_state

# DATE = str(datetime.datetime.today())
# LOG_FILE = os.path.join('/home/mara/Desktop/logs/A2C_OAI_NENVS', DATE)

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
                 ent_coef, vf_coef, max_grad_norm, lr, alpha, epsilon,
                 units_per_hlayer, total_timesteps, lrschedule, log_interval, logdir, keep_model, activ_fcn):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up A2C learning agent')
        self.logger.info('nsteps: ' + str(total_timesteps))
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        eval_model = policy(sess, ob_space, ac_space, 1, 1, units_per_hlayer, reuse=False, activ_fcn=activ_fcn)
        step_model = policy(sess, ob_space, ac_space, nenvs, 1, units_per_hlayer, reuse=tf.AUTO_REUSE, activ_fcn=activ_fcn)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)

        # Compute cross entropy loss between estimated distribution of action and 'true' distribution of actions
        chosen_action_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi_logit, labels=A)

        pg_loss = tf.reduce_mean(ADV * chosen_action_log_probs) # minimize
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))  # minimize
        entropy = - tf.reduce_mean(cat_entropy(train_model.pi_logit))  # maximize
        loss = pg_loss + entropy*ent_coef + vf_loss * vf_coef

        vf = tf.squeeze(train_model.vf)

        # pg_loss = tf.reduce_sum(ADV * neglogpac)
        # entropy = -tf.reduce_sum(tf.nn.softmax(train_model.pi) * tf.nn.log_softmax(train_model.pi))
        # pg_loss -= ent_coef * entropy
        #
        # # Estimate the value loss using the sum of squared errors.
        # vf_loss = tf.nn.l2_loss(train_model.vf - R)
        # loss = pg_loss + 0.5 * vf_loss

        params = find_trainable_variables("model")
        trainer = tf.train.AdamOptimizer(learning_rate=LR)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        gradients = trainer.compute_gradients(loss)
        grads, variables = zip(*gradients)
        # grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        _train = [trainer.apply_gradients(grads),
                  self.global_step.assign_add(nbatch)]

        if log_interval > 0:
            for g, v in gradients:
                if g is not None:
                    tf.summary.histogram("%s-grad" % v.name.replace(':', '_'), g)
            for p in params:
                if p is not None:
                    tf.summary.histogram("train/%s" % p.name.replace(':', '_'), p.value())
            tf.summary.scalar("train/pg_loss", pg_loss)
            tf.summary.scalar("train/vf_loss", vf_loss)
            tf.summary.scalar("train/entropy", entropy)
            tf.summary.histogram("others/ADV", ADV)
            tf.summary.histogram("others/neglocpac", chosen_action_log_probs)
            tf.summary.histogram("others/vf", vf)
            self.summary_step = tf.summary.merge_all()

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        # Adding these to collection so we can restore them again
        tf.add_to_collection('inputs', eval_model.X)
        tf.add_to_collection('pi', eval_model.pi)
        tf.add_to_collection('pi_logit', eval_model.pi_logit)
        tf.add_to_collection('val', eval_model.vf)
        tf.add_to_collection('step', eval_model.ac)
        try:
            add_to_collection_rnn_state('state_in', eval_model.rnn_state_in)
            add_to_collection_rnn_state('state_out', eval_model.rnn_state_out)
        except AttributeError:
            pass

        tf.global_variables_initializer().run(session=sess)

        def train(obs, states, rewards, actions, values):
            advs = rewards - values  # Estimate for A = Q(s,a) - V(s)
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.rnn_state_in] = states
                # td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, test, ap, global_step = sess.run(
                [pg_loss, vf_loss, entropy, _train, train_model.pi, self.global_step],
                td_map
            )
            # TF summary logging
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                self.logger.info('Save summary of network weights, grads and losses.')
                summary_str = sess.run(self.summary_step, td_map)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)

            # In config evaluation mode save model regularly to track performance of the learning agent.
            # if save_interval > 0 and (self.num_steps_trained % save_interval == 0):
            #     self.logger.info('Save model at train step %s' % self.num_steps_trained)
            #     self.save('config_model')
            #     eval = True
            # else:
            #     eval = False

                # run evaluation step

                # TODO add self.save('inter_model') here ( every N trai steps) -> fro config evaluation
                # it's good to use interaction step still as idx to have this infromation for later analysis.
            self.num_steps_trained += 1

            return policy_loss, value_loss, policy_entropy, ap

        saver = tf.train.Saver(max_to_keep=7)
        # saver = tf.train.Saver(max_to_keep=keep_model)

        def save(f_name):
            # test_run(20)
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
            for i in range(0, n_eps):
                obs = env.reset()
                obs = normalize_obs(obs)
                done = False
                total_return = 0
                total_length = 0

                while not done and (total_return < n_pipes):
                    pi, pi_log, act = sess.run([eval_model.pi, eval_model.pi_logit, eval_model.ac], feed_dict={eval_model.X: [obs]})
                    ac = np.argmax(pi_log)
                    obs, reward, done, _ = env.step(ac)
                    # obs, reward, done, _ = env.step(act[0][0])
                    obs = normalize_obs(obs)
                    total_length += 1
                    total_return += reward
                # logger.debug('*****************************************')
                self.logger.info('Episode %s: %s, %s' % (i, total_return, total_length))
                ep_length.append(total_length)
                ep_return.append(total_return)
            return ep_return

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        self.test_run = test_run

        # Set the summary writer to write to the given logdir if logging is enabled
        if log_interval > 0:
            self.summary_writer = tf.summary.FileWriter(logdir, graph_def=sess.graph_def)
        else:
            self.summary_writer = None

    def get_summary_writer(self):
        return self.summary_writer


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, show_interval=0, summary_writer=None):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up %s-step Runner' % nsteps)
        self.env = env
        self.model = model
        nd, = env.observation_space.shape
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nd)
        self.obs = np.zeros((nenv, nd), dtype=np.float32)
        self.nc = None  # nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        # stats
        self.eplength = [0 for _ in range(nenv)]
        self.epreturn = [0 for _ in range(nenv)]
        # self.retbuffer = deque(maxlen=1*nenv)
        # self.avg_return_n_episodes = 0
        self.max_return = 40  # threshold below which no model will be saved.

        # rendering
        self.show_interval = show_interval
        self.ep_idx = [0 for _ in range(nenv)]

        self.summary_writer = summary_writer

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states)  # , self.dones) ?
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)

            rewards = [rewards[i] - 1e-5 for i in range(len(rewards))]
            obs = normalize_obs(obs)
            self.logger.debug('Observations: %s' % obs)

            # render only every i-th episode
            if self.show_interval != 0:
                if (self.ep_idx[0] % self.show_interval) == 0:
                    self.env.render()

            self.eplength = [self.eplength[i] + 1 for i in range(self.nenv)] # Todo use already implemented functions in run_ple_utils!!!
            self.epreturn = [self.epreturn[i] + rewards[i] for i in range(self.nenv)]

            # Check for terminal states in every env
            for i, done in enumerate(dones):  # i -> environment ID
                if done:
                    self.ep_idx[i] += 1
                    self.obs[i] = self.obs[i]*0

                    # update tensorboard summary
                    if self.summary_writer is not None:
                        summary = tf.Summary()
                        summary.value.add(tag='envs/environment%s/episode_length' % i,
                                          simple_value=self.eplength[i])
                        summary.value.add(tag='envs/environment%s/episode_reward' % i,
                                          simple_value=self.epreturn[i])
                        self.summary_writer.add_summary(summary, self.ep_idx[i])  #self.global_step.eval())
                        self.summary_writer.flush()
                    # self.retbuffer.append(self.epreturn[i])
                    if self.epreturn[i] > self.max_return:
                        self.max_return = self.epreturn[i]
                        self.logger.info('Save model at max reward %s' % self.max_return)
                        self.model.save('inter_model')
                    self.eplength[i] = 0
                    self.epreturn[i] = 0

            # Reset RNN state vector to 0 if previous sample is a terminating one.
            # As no history should be used in rnn training then.
            if states:
                env_was_done = False
                for i, done in enumerate(self.dones):
                    if done and not env_was_done:
                        env_was_done = True
                        c_new = states[0]
                        h_new = states[1]
                        c_new[i] = np.zeros_like(c_new[i])
                        h_new[i] = np.zeros_like(h_new[i])
                    elif done:
                        c_new[i] = np.zeros_like(c_new[i])
                        h_new[i] = np.zeros_like(h_new[i])
                if env_was_done:
                    states = tf.contrib.rnn.LSTMStateTuple(c_new, h_new)
                    # print(states)
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        # mb_masks = mb_dones[:, :-1] ?
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states).tolist()  # , self.dones).tolist()  ?
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            self.logger.debug('Discounted rewards: %s' % rewards)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        # mb_masks = mb_masks.flatten()  ?

        # if len(self.retbuffer) > 0:
        #     self.avg_return_n_episodes = np.mean(self.retbuffer)

        self.logger.debug('Actions: %s' % mb_actions)
        self.logger.debug('Q values: %s' % mb_values)
        # self.logger.debug('Done mask: %s' % mb_masks)  # ?
        self.logger.debug('Observations: %s' % mb_obs)

        return mb_obs, mb_states, mb_rewards, mb_actions, mb_values, self.max_return  # self.avg_return_n_episodes
        # return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, self.max_return  # self.avg_return_n_episodes  ?

def learn(policy, env, test_env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
          units_per_hlayer=None, log_interval=100, logdir='/mnt/logs/A2C', keep_model=7, activ_fcn='relu6',
          test_interval=0, show_interval=0):
    logger = logging.getLogger(__name__)
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy,
                  ob_space=ob_space,
                  ac_space=ac_space,
                  nenvs=nenvs,
                  nsteps=nsteps,
                  ent_coef=ent_coef,
                  vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr,
                  alpha=alpha, # TODO remove
                  epsilon=epsilon, # TODO remove
                  units_per_hlayer=units_per_hlayer,
                  total_timesteps=total_timesteps,
                  lrschedule=lrschedule,
                  log_interval=log_interval,
                  logdir=logdir,
                  keep_model=keep_model,
                  activ_fcn=activ_fcn)

    sum_write = model.get_summary_writer()
    result_path = os.path.join(logdir, 'results.csv')

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, show_interval=show_interval, summary_writer=sum_write)
    i_sample, i_train = 0, 0
    rolling_mean = deque(maxlen=30)

    logger.info('Start Training')
    nbatch = nenvs*nsteps
    tstart = time.time()
    max_avg_ep_return = -5  # moving average of 20*nenv training episodes
    max_returns = deque([50], maxlen=7)  # returns of the 7 best training episodes
    for update in range(1, total_timesteps//nbatch + 1):
        # obs, states, rewards, masks, actions, values, avg_ep_return = runner.run()
        # policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, masks, actions, values)
        obs, states, rewards, actions, values, avg_ep_return = runner.run()
        policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, actions, values)
        if test_interval > 0 and (update % test_interval == 0):
            ep_return = model.test_run(test_env, n_eps=30, n_pipes=2000)
            with open(result_path, "a") as csvfile:
                writer = csv.writer(csvfile)
                ep_return = [str(p) for p in ep_return]
                ep_return.insert(0, ('step_%s' % i_sample))
                writer.writerow(ep_return)
        i_sample += nbatch
        i_train += 1

        # nseconds = time.time()-tstart
        # fps = int((update*nbatch)/nseconds)
        # if update % log_interval == 0 or update == 1:
        #     ev = explained_variance(values, rewards)
        #     logger.record_tabular("nupdates", update)
        #     logger.record_tabular("total_timesteps", update*nbatch)
        #     logger.record_tabular("fps", fps)
        #     logger.record_tabular("policy_entropy", float(policy_entropy))
        #     logger.record_tabular("value_loss", float(value_loss))
        #     logger.record_tabular("explained_variance", float(ev))
        #     logger.dump_tabular()

    model.save('final_model')
    logger.info('Finished Training. Saving Final model.')

    logger.info('*******************************************************')
    logger.info('Total number of interactions with the environment: %s' % i_sample)
    logger.info('Total number of finished episodes during training: sum(%s) = %s' % (runner.ep_idx, sum(runner.ep_idx)))
    logger.info('Total number of parameter updates during training: %s' % i_train)
    logger.info('*******************************************************\n')


from run_ple_utils import make_ple_envs, make_ple_env  #, arg_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy, LargerLSTMPolicy
if __name__ == '__main__':
    seed = 2
    # env = make_ple_envs('FlappyBird-v2', num_env=3, seed=seed,
    #                     trace_length=500, offset=1., amplitude=0.1, fband=[0.0001, 0.005],  # Filtered Random Walk
    #                     nsteps=20, time_interval=[20,200], value_interval=[3,6])            # Random Steps
    env = make_ple_envs('ContFlappyBird-v2', num_env=3, seed=seed)
    test_env = make_ple_env('FlappyBird-v2', seed=seed)
    logger = logging.getLogger()
    ch = logging.StreamHandler()  # Handler which writes to stderr (in red)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    logdir = '/home/mara/Desktop/logs/A2C_OAI_NENVS'
    logdir = os.path.join(logdir, datetime.datetime.today().strftime('%Y_%m_%d_%H%M%S'))

    # learn(LargerMLPPolicy, env,
    learn(LargerLSTMPolicy, env, test_env=test_env,
          seed=seed,
          nsteps=50,
          vf_coef=0.2,
          ent_coef=1e-7,
          gamma=0.90,
          lr=5e-4,
          lrschedule='constant',
          max_grad_norm=0.01,
          log_interval=30,
          test_interval=10,
          show_interval=0,
          units_per_hlayer=(64,64,64),
          total_timesteps=40000,  # int(1e7),
          logdir=logdir,
          activ_fcn='relu6')
    env.close()
