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

import matplotlib.pyplot as plt

# DATE = str(datetime.datetime.today())
# LOG_FILE = os.path.join('/home/mara/Desktop/logs/A2C_OAI_NENVS', DATE)

class Model(object):

    def __init__(self, policy, ob_space, ac_space,
                 lr, max_grad_norm, units_per_hlayer, activ_fcn, log_interval, logdir,
                 nenvs, batch_size, ent_coef, vf_coef, keep_model, meta=False):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up A2C learning agent')
        # self.logger.info('nsteps: ' + str(total_timesteps))
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()
        nact = ac_space.n
        nbatch = nenvs*batch_size

        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)

        eval_model = policy(sess, ob_space, ac_space, 1, 1, units_per_hlayer, reuse=False, activ_fcn=activ_fcn)
        step_model = policy(sess, ob_space, ac_space, nenvs, 1, units_per_hlayer, reuse=tf.AUTO_REUSE, activ_fcn=activ_fcn)
        train_model = policy(sess, ob_space, ac_space, nenvs, batch_size, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)

        # -- Loss computation --
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        # LR = tf.placeholder(tf.float32, [])

        def get_loss(model, placeholder_dict):
            a = placeholder_dict["A"]
            adv = placeholder_dict["ADV"]
            r = placeholder_dict["R"]
            # Compute cross entropy loss between estimated distribution of action and 'true' distribution of actions
            chosen_action_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.pi_logit,
                                                                                     labels=a)
            pg_loss = tf.reduce_mean(adv * chosen_action_log_probs)  # minimize
            vf_loss = tf.reduce_mean(mse(tf.squeeze(model.vf), r))  # minimize
            entropy = - tf.reduce_mean(cat_entropy(model.pi_logit))  # maximize
            return pg_loss, entropy, vf_loss, model.vf, chosen_action_log_probs, None, None

        # # Compute cross entropy loss between estimated distribution of action and 'true' distribution of actions
        # chosen_action_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi_logit, labels=A)
        #
        # pg_loss = tf.reduce_mean(ADV * chosen_action_log_probs) # minimize
        # vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))  # minimize
        # entropy = - tf.reduce_mean(cat_entropy(train_model.pi_logit))  # maximize
        self.input_plchld = {'A': A, 'ADV': ADV, 'R': R}
        pg_loss, entropy, vf_loss, _, chosen_action_log_probs, _, _ = get_loss(train_model, self.input_plchld)
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
        # trainer = tf.train.AdamOptimizer(learning_rate=LR)
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
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

        # lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        # Adding these to collection so we can restore them again
        tf.add_to_collection('inputs', eval_model.X)
        tf.add_to_collection('pi', eval_model.pi)
        tf.add_to_collection('pi_logit', eval_model.pi_logit)
        tf.add_to_collection('val', eval_model.vf)
        tf.add_to_collection('step', eval_model.ac)
        if eval_model.initial_state is not None:
            add_to_collection_rnn_state('state_in', eval_model.rnn_state_in)
            add_to_collection_rnn_state('state_out', eval_model.rnn_state_out)

        tf.global_variables_initializer().run(session=sess)

        def train(obs, states, rewards, actions, values):
            advs = rewards - values  # Estimate for A = Q(s,a) - V(s)
            # for step in range(len(obs)):
            #     cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards} #, LR:cur_lr}
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

        # saver = tf.train.Saver(max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=keep_model)

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
            self.logger.info('Evaluating current agent')
            ep_return = []
            ep_length = []
            for i in range(0, n_eps):
                obs = env.reset()
                obs = normalize_obs(obs)
                done = False
                if eval_model.initial_state is not None:
                    if len(eval_model.initial_state) > 1:
                        rnn_s_in = (np.zeros(eval_model.initial_state[0].shape), np.zeros(eval_model.initial_state[1].shape))  # init lstm cell vector
                    else:
                        rnn_s_in = np.zeros(eval_model.initial_state.shape)  # init gru cell vector
                    # rnn_s_in = (np.zeros([1, units_per_hlayer[1]]), np.zeros([1, units_per_hlayer[1]]))
                total_return = 0
                total_length = 0
                # self.logger.info('Eps %s' % i)

                while not done and (total_return < n_pipes):
                    # self.logger.info(total_return)
                    if eval_model.initial_state is not None:
                        # self.logger.info('here')
                        pi, pi_log, act, rnn_s_out = sess.run([eval_model.pi, eval_model.pi_logit, eval_model.ac, eval_model.rnn_state_out], feed_dict={eval_model.X: [obs], eval_model.rnn_state_in: rnn_s_in})
                        # self.logger.info('there')
                    else:
                        # self.logger.info('here1')
                        pi, pi_log, act = sess.run([eval_model.pi, eval_model.pi_logit, eval_model.ac], feed_dict={eval_model.X: [obs]})
                        # self.logger.info('there1')

                    ac = np.argmax(pi_log)
                    # self.logger.info('take action %s' % ac)
                    obs, reward, done, _ = env.step(ac)
                    # self.logger.info('stepped')
                    # obs, reward, done, _ = env.step(act[0][0])
                    obs = normalize_obs(obs)
                    total_length += 1
                    total_return += reward
                    if eval_model.initial_state is not None:
                        rnn_s_in = rnn_s_out

                # logger.debug('*****************************************')
                self.logger.info('Episode %s: %s, %s' % (i, total_return, total_length))
                ep_length.append(total_length)
                ep_return.append(total_return)
            return ep_return

        self.train_vars = params
        self.train = train
        self.train_model = train_model
        self.eval_model = eval_model
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

        self.sess = sess

    def get_summary_writer(self):
        return self.summary_writer


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, horizon=100, show_interval=0, summary_writer=None):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up %s-step Runner' % nsteps)
        self.env = env
        self.model = model
        nd, = env.observation_space.shape
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nd)
        self.obs = np.zeros((nenv, nd), dtype=np.float32)  # why do we initialize it with 0?
        self.obs[:] = env.reset()
        # self.nc = None  # nc
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        # stats
        self.eplength = [0 for _ in range(nenv)]
        self.epreturn = [0 for _ in range(nenv)]
        self.reward_window = [deque(maxlen=horizon) for _ in range(nenv)]
        # self.retbuffer = deque(maxlen=1*nenv)
        # self.avg_return_n_episodes = 0
        self.return_threshold = -2.  # 0.0  # 40  # threshold below which no model will be saved.

        # rendering
        self.show_interval = show_interval
        self.ep_idx = [0 for _ in range(nenv)]

        self.summary_writer = summary_writer

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_rawrewards = [],[],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, pi, values, states, _ = self.model.step(self.obs, self.states)  # , self.dones) ?
            # print(pi)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # print(obs)
            # rewards = [rewards[i] - 1e-5 for i in range(len(rewards))]
            obs = normalize_obs(obs)
            # print(obs)
            self.logger.debug('Observations: %s' % obs)

            # render only every i-th episode
            if self.show_interval != 0:
                if (self.ep_idx[0] % self.show_interval) == 0:
                    self.env.render()

            self.eplength = [self.eplength[i] + 1 for i in range(self.nenv)] # Todo use already implemented functions in run_ple_utils!!!
            self.epreturn = [self.epreturn[i] + rewards[i] for i in range(self.nenv)]
            [self.reward_window[i].append(rewards[i]) for i in range(self.nenv)]

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
                    if self.epreturn[i] > self.return_threshold:
                        self.return_threshold = self.epreturn[i]
                        self.logger.info('Save model at max reward %s' % self.return_threshold)
                        self.model.save('inter_model')
                    self.eplength[i] = 0
                    self.epreturn[i] = 0

            # # Is not necessary, as the environment is continuous now!
            # # Reset RNN state vector to 0 if previous sample is a terminating one.
            # # As no history should be used in rnn training then.
            # if states:
            #     env_was_done = False
            #     for i, done in enumerate(self.dones):
            #         if done and not env_was_done:
            #             env_was_done = True
            #             c_new = states[0]
            #             h_new = states[1]
            #             c_new[i] = np.zeros_like(c_new[i])
            #             h_new[i] = np.zeros_like(h_new[i])
            #         elif done:
            #             c_new[i] = np.zeros_like(c_new[i])
            #             h_new[i] = np.zeros_like(h_new[i])
            #     if env_was_done:
            #         states = tf.contrib.rnn.LSTMStateTuple(c_new, h_new)
            #         # print(states)
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
        mb_rawrewards = np.copy(mb_rewards)
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

        return mb_obs, mb_states, mb_rewards, mb_actions, mb_values, self.reward_window, mb_rawrewards  # self.avg_return_n_episodes
        # return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, self.max_return  # self.avg_return_n_episodes  ?

def learn(policy, env, test_env, seed, total_timesteps,
          log_interval, test_interval, show_interval, logdir,
          lr, max_grad_norm, units_per_hlayer, activ_fcn,
          gamma=0.99, vf_coef=0.5, ent_coef=0.01, batch_size=5,
          early_stop=False, keep_model=2,
          save_model=True, restore_model=False, save_traj=False):
    logger = logging.getLogger(__name__)
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy,
                  ob_space=ob_space,
                  ac_space=ac_space,
                  lr=lr,
                  max_grad_norm=max_grad_norm,
                  activ_fcn=activ_fcn,
                  units_per_hlayer=units_per_hlayer,
                  log_interval=log_interval,
                  logdir=logdir,

                  nenvs=nenvs,
                  batch_size=batch_size,
                  ent_coef=ent_coef,
                  vf_coef=vf_coef,
                  keep_model=keep_model
                  # total_timesteps=total_timesteps,
                  )

    sum_write = model.get_summary_writer()
    result_path = os.path.join(logdir, 'train_results.csv')
    if save_traj:
        rew_traj = []
        rew_results_path = os.path.join(logdir, ('lr'+str(lr)+'_tracking_results.csv'))
    else:
        rew_results_path = None

    i_sample, i_train = 0, 0
    return_threshold = -0.05
    horizon = 100
    avg_rm = deque(maxlen=30)

    runner = Runner(env, model, nsteps=batch_size, gamma=gamma, horizon=horizon, show_interval=show_interval, summary_writer=sum_write)

    if restore_model:
        for el in os.listdir(logdir):
            if 'final' in el and '.meta' in el:
                # Load pre trained model and set network parameters
                model.load(os.path.join(logdir, el[:-5]))
                # Reset global step parameter.
                model.sess.run(model.global_step.assign(0))

    logger.info('Start Training')
    breaked = False
    nbatch = nenvs*batch_size
    tstart = time.time()
    # max_avg_ep_return = -5  # moving average of 20*nenv training episodes
    max_returns = deque([50], maxlen=7)  # returns of the 7 best training episodes
    for update in range(1, total_timesteps//nbatch + 1):
        # obs, states, rewards, masks, actions, values, avg_ep_return = runner.run()
        # policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, masks, actions, values)
        obs, states, rewards, actions, values, reward_window, raw_rewards = runner.run()
        if rew_results_path is not None:
            rew_traj.append(raw_rewards)
        # print('\nMEAN:%s\n' % np.mean(obs, axis=0))
        # plt.figure()
        # d = [o *512 for o  in obs[:,0]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 7 for o in obs[:, 1]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 2]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 3]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 4]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 5]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 6]]
        # plt.plot(d)
        # plt.figure()
        # d = [o * 512 for o in obs[:, 7]]
        # plt.plot(d)
        # plt.show()
        policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, actions, values)
        if test_interval > 0 and i_train > 0 and (update % test_interval == 0):
            ep_return = model.test_run(test_env, n_eps=10, n_pipes=2000)  # TODO test, whether results.csv is saved properly
            with open(result_path, "a") as csvfile:
                writer = csv.writer(csvfile)
                # ep_return = [str(p) for p in ep_return]
                # ep_return.insert(0, ('step_%s' % i_sample))
                ep_return[0:0] = [i_sample, i_train]
                writer.writerow(ep_return)

        # Log the performance during training at every update step.
        # Save the current model if the average reward of the last
        # 100 time steps is above the return threshold
        if ('ContFlappyBird' in env.env_id):
            saved = False
            for i, rw in enumerate(reward_window):
                rm = sum(rw) / horizon
                if sum_write is not None:
                    s_summary = tf.Summary()
                    s_summary.value.add(tag='envs/environment%s/isample_return' % i,
                                        simple_value=rm)
                    sum_write.add_summary(s_summary, i_sample)

                    t_summary = tf.Summary()
                    t_summary.value.add(tag='envs/environment%s/itrain_return' % i,
                                        simple_value=rm)
                    sum_write.add_summary(t_summary, i_train)
                    sum_write.flush()
                # logger.info(rm)
                if save_model and not saved and rm > return_threshold:
                    return_threshold = rm
                    logger.info('Save model at max rolling mean %s' % return_threshold)
                    model.save('inter_model')
                    saved = True
                avg_rm.append(rm)

        if early_stop:
            if (i_sample > 500000) and (i_sample <= 500000 + nbatch):  # TODO how to determine early-stopping criteria non-heuristically, but automatically? - BOHB algorithm?
                if (sum(avg_rm)/30) <= -0.88:
                    print('breaked')
                    breaked = True
                    break
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

    if save_model:
        model.save('final_model')
        logger.info('Finished Training. Saving Final model.')

    if rew_results_path is not None:
        with open(rew_results_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            traj = np.asanyarray(rew_traj).reshape(-1).tolist()
            traj[0:0] = [np.mean(traj)]  # i_train, i_sample
            writer.writerow(traj)

    logger.info('*******************************************************')
    logger.info('Total number of interactions with the environment: %s' % i_sample)
    logger.info('Total number of finished episodes during training: sum(%s) = %s' % (runner.ep_idx, sum(runner.ep_idx)))
    logger.info('Total number of parameter updates during training: %s' % i_train)
    logger.info('*******************************************************\n')

    return breaked


from run_ple_utils import make_ple_envs, make_ple_env  #, arg_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy, LargerLSTMPolicy
if __name__ == '__main__':
    seed = 0
    # env = make_ple_envs('FlappyBird-v2', num_env=3, seed=seed,
    #                     trace_length=500, offset=1., amplitude=0.1, fband=[0.0001, 0.005],  # Filtered Random Walk
    #                     nsteps=20, time_interval=[20,200], value_interval=[3,6])            # Random Steps
    # env = make_ple_envs('ContFlappyBird-v1', num_env=4, seed=seed)
    # test_env = make_ple_env('ContFlappyBird-v3', seed=seed)

    seed = 1
    env = make_ple_envs('ContFlappyBird-hNS-nrf0-train-v0', num_env=1, seed=seed)
    test_env = make_ple_env('ContFlappyBird-v3', seed=seed)
    logger = logging.getLogger()
    ch = logging.StreamHandler()  # Handler which writes to stderr (in red)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    BATCH_SIZE = 64

    # SMAC config 1
    ACTIV_FCN = 'mixed'
    DISCOUNT = 0.94
    ENT_COEFF = 0.000036
    VF_COEFF = 0.36
    LR = 0.0032
    P_LAYER = 17
    S_LAYER1 = 78
    S_LAYER2 = 35

    MAX_REPLAY_BUF_SIZE = 500
    BATCH_SIZE = 64
    MAX_GRAD_NORM = 0.1
    NUM_TRAIN_UPDATES = int(2e3)
    LOG_INTERVAL = 30

    logdir = '/home/mara/Desktop/logs/LSTM_A2C-lala'
    # logdir = os.path.join(logdir, datetime.datetime.today().strftime('%Y_%m_%d_%H%M%S'))
    logdir = os.path.join(logdir, 'a2c_output1')
    # learn(LargerMLPPolicy, env,
    learn(LargerLSTMPolicy,
          env=env,
          test_env=test_env,
          seed=seed,
          total_timesteps=NUM_TRAIN_UPDATES,  # int(1e7),
          log_interval=LOG_INTERVAL,
          test_interval=0,
          show_interval=0,
          logdir=logdir,
          lr=LR,
          # lrschedule='constant',
          max_grad_norm=MAX_GRAD_NORM,
          units_per_hlayer=(S_LAYER1, S_LAYER2, P_LAYER),
          activ_fcn=ACTIV_FCN,
          gamma=DISCOUNT,
          vf_coef=VF_COEFF,
          ent_coef=ENT_COEFF,
          batch_size=BATCH_SIZE,
          save_traj=True)
    env.close()
