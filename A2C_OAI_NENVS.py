import os
import os.path as osp
import gym
import time
import datetime
import joblib
import logging
import numpy as np
import tensorflow as tf
import logger
from collections import deque
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session, set_global_seeds
from utils_OAI import cat_entropy, mse, explained_variance, normalize_obs
from utils_OAI import discount_with_dones

# DATE = str(datetime.datetime.today())
# LOG_FILE = os.path.join('/home/mara/Desktop/logs/A2C_OAI_NENVS', DATE)

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, units_per_hlayer=None, total_timesteps=int(80e6), lrschedule='linear',
            log_interval=100, logdir=''):
        print('logdir is:' + logdir)
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

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, units_per_hlayer, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, units_per_hlayer, reuse=True)


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
                    tf.summary.histogram("%s-grad" % v.name, g)
            for p in params:
                if p is not None:
                    tf.summary.histogram("train/%s" % p.name, p.value())
            tf.summary.scalar("train/pg_loss", pg_loss)
            tf.summary.scalar("train/vf_loss", vf_loss)
            tf.summary.scalar("train/entropy", entropy)
            tf.summary.histogram("others/ADV", ADV)
            tf.summary.histogram("others/neglocpac", chosen_action_log_probs)
            tf.summary.histogram("others/vf", vf)
            self.summary_step = tf.summary.merge_all()

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        saver = tf.train.Saver(max_to_keep=7)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _, ap, global_step = sess.run(
                [pg_loss, vf_loss, entropy, _train, train_model.pi, self.global_step],
                td_map
            )
            # TF summary logging
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                print('saved tf summary')
                summary_str = sess.run(self.summary_step, td_map)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)
            self.num_steps_trained += 1

            return policy_loss, value_loss, policy_entropy, ap

        def save(save_path, f_name):
            gs = sess.run(self.global_step)
            saver.save(sess, os.path.join(save_path, f_name), global_step=gs)
            # ps = sess.run(params)
            # make_path(save_path)
            # joblib.dump(ps, os.path.join(save_path, f_name))

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

        # Set the summary writer to write to the given logdir if logging is enabled
        if log_interval > 0:
            self.summary_writer = tf.summary.FileWriter(logdir, graph_def=sess.graph_def)
        else:
            self.summary_writer = None

    def get_summary_writer(self):
        return self.summary_writer


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, show_interval=0, summary_writer=None):
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
        self.retbuffer = deque(maxlen=20*nenv)
        self.max_return = 0

        # rendering
        self.show_interval = show_interval
        self.ep_idx = [0 for _ in range(nenv)]

        self.summary_writer = summary_writer

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # obs, rewards, dones, _ = self.env.step(actions)

            rewards = [rewards[i] - 1e-5 for i in range(len(rewards))]
            obs = normalize_obs(obs)
            # print(obs)

            # render only every i-th episode
            if self.show_interval != 0:
                if sum(self.ep_idx) % self.show_interval == 0:
                    self.env.render()

            self.eplength = [self.eplength[i] + 1 for i in range(self.nenv)]
            self.epreturn = [self.epreturn[i] + rewards[i] for i in range(self.nenv)]
            self.states = states
            self.dones = dones

            if self.retbuffer:
                avg_return_n_episodes = np.mean(self.retbuffer)

            # Check for terminal states in every env
            for i, done in enumerate(dones):
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
                        # summary.value.add(tag='environment/fps',
                        #                   simple_value=self.env.episode_length / self.env.episode_run_time)
                        self.summary_writer.add_summary(summary, self.ep_idx[i])  #self.global_step.eval())
                        self.summary_writer.flush()
                    self.retbuffer.append(self.epreturn[i])
                    if self.epreturn[i] > self.max_return: self.max_return = self.epreturn[i]
                    self.eplength[i] = 0
                    self.epreturn[i] = 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, avg_return_n_episodes

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
          units_per_hlayer=None, log_interval=100, logdir='/mnt/logs/A2C',
          save_interval=1000, show_interval=0):
    # TODO only use defaults values for arguments once and not again in model to ensure that default value is set only once.
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
                  vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon,
                  units_per_hlayer=units_per_hlayer, total_timesteps=total_timesteps,
                  lrschedule=lrschedule, log_interval=log_interval, logdir=logdir)
    sum_write = model.get_summary_writer()
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, show_interval=show_interval, summary_writer=sum_write)

    nbatch = nenvs*nsteps
    tstart = time.time()
    max_avg_ep_return = -5
    for update in range(1, total_timesteps//nbatch + 1):
        obs, states, rewards, masks, actions, values, avg_ep_return = runner.run()
        policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, masks, actions, values)
        # print(ap)

        # Save network model if average return of last 20 * nenvs episodes is higher than threshold 30
        # and at least 5 points higher than the recent maximum average return per 20 * nenvs episodes.
        print(avg_ep_return)
        if avg_ep_return > 0 and avg_ep_return >= (max_avg_ep_return + 5):
            max_avg_ep_return = avg_ep_return
            model.save(logdir, 'inter_model', )
        # if update % save_interval == 0:
        #     model.save(logdir, 'inter_model', )  # save model not w.r.t to training step but depending on achieved result.
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            # logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            # logger.record_tabular("fps", fps)
            # logger.record_tabular("policy_entropy", float(policy_entropy))
            # logger.record_tabular("value_loss", float(value_loss))
            # logger.record_tabular("explained_variance", float(ev))
            # logger.dump_tabular()
    model.save(logdir, 'final_model')

    # return np.mean(runner.retbuffer), np.var(runner.retbuffer), runner.max_return

from run_ple_utils import make_ple_envs  #, arg_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy
if __name__ == '__main__':
    seed = 2
    env = make_ple_envs('FlappyBird-v1', num_env=3, seed=seed)
    logdir = '/home/mara/Desktop/logs/A2C_OAI_NENVS'
    logdir = os.path.join(logdir, str(datetime.datetime.today()))
    learn(LargerMLPPolicy, env,
           seed=seed,
           nsteps=50,
           vf_coef=0.2,
           ent_coef=1e-7,
           gamma=0.90,
           lr=5e-4,
           lrschedule='constant',
           max_grad_norm=0.01,
           log_interval=30,
           save_interval=1000,
           show_interval=30,
           units_per_hlayer=(32,32,32),
           total_timesteps=40000,  # int(1e7),
           logdir=logdir)

    # print('average performance: %s' % avg_perf)
    # print('performance variance: %s' % var_perf)
    # print('maximal return was: %s' % max_return)
    env.close()
