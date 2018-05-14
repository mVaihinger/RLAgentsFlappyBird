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

from models_OAI import MlpPolicy, FCPolicy, CastaPolicy
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session, set_global_seeds
from utils_OAI import cat_entropy, mse, explained_variance
from utils_OAI import discount_with_dones

DATE = str(datetime.datetime.today())
LOG_FILE = os.path.join('/home/mara/Desktop/logs/A2C_OAI_NENVS', DATE)

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            log_interval=100):

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

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)



        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)

        pg_loss = tf.reduce_mean(ADV * neglogpac) # minimize
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))  # minimize
        entropy = - tf.reduce_mean(cat_entropy(train_model.pi))  # maximize
        loss = pg_loss + entropy*ent_coef + vf_loss * vf_coef

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

        for g, v in gradients:
            if g is not None:
                tf.summary.histogram("%s-grad" % v.name, g)
        for p in params:
            if p is not None:
                tf.summary.histogram("train/%s" % p.name, p.value())
        tf.summary.scalar("train/pg_loss", pg_loss)
        tf.summary.scalar("train/vf_loss", vf_loss)
        tf.summary.scalar("train/entropy", entropy)
        self.summary_step = tf.summary.merge_all()

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

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
            if self.num_steps_trained % self.log_interval == 0:
            # if log_interval != 0 and self.num_steps_trained % self.log_interval == 0:
                summary_str = sess.run(self.summary_step, td_map)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), global_step)

            self.num_steps_trained += 1

            return policy_loss, value_loss, policy_entropy, ap

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

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

        # Set the logs writer to the folder /tmp/tensorflow_logs
        self.summary_writer = tf.summary.FileWriter(LOG_FILE, graph_def=sess.graph_def)

    def get_summary_writer(self):
        return self.summary_writer


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, show_interval=0, summary_writer=None):
        self.env = env
        self.model = model
        nd, = env.observation_space.shape
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nd)
        self.obs = np.zeros((nenv, nd), dtype=np.uint8)
        self.nc = None  # nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        # stats
        self.eplength = [0 for _ in range(nenv)]
        self.epreturn = [0 for _ in range(nenv)]

        # rendering
        self.show_interval = show_interval
        self.ep_idx = 0

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

            # render only every i-th episode
            if self.show_interval != 0:
                if self.ep_idx % self.show_interval == 0:
                    env.render()

            self.eplength = [self.eplength[i] + 1 for i in range(self.nenv)]
            self.epreturn = [self.epreturn[i] + rewards[i] for i in range(self.nenv)]
            self.states = states
            self.dones = dones

            for n, done in enumerate(dones):
                if done:
                    if n == 0:
                        self.ep_idx += 1
                    self.obs[n] = self.obs[n]*0  # TODO how do they do this her in the paper?
                    # update tensorboard summary
                    if self.summary_writer is not None:
                        summary = tf.Summary()
                        summary.value.add(tag='envs/environment%s/episode_length' % n,
                                          simple_value=self.eplength[n])
                        summary.value.add(tag='envs/environment%s/episode_reward' % n,
                                          simple_value=self.epreturn[n])
                        # summary.value.add(tag='environment/fps',
                        #                   simple_value=self.env.episode_length / self.env.episode_run_time)
                        self.summary_writer.add_summary(summary, self.ep_idx)  #self.global_step.eval())
                        self.summary_writer.flush()

                    # print('env %s ended after %s steps with return %s' % (n, self.eplength[n], self.epreturn[n]))
                    self.eplength[n] = 0
                    self.epreturn[n] = 0
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
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, display_screen=False):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    sum_write = model.get_summary_writer()
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, show_interval=30, summary_writer=sum_write)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, masks, actions, values)
        print(ap)
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
    env.close()


from run_ple_utils import make_ple_env
if __name__ == '__main__':

    env = make_ple_env('FlappyBird-v1', num_env=3, seed=0)
    learn(CastaPolicy, env, 0, nsteps=50, vf_coef=0.2, ent_coef=1e-7, gamma=0.90, display_screen=True,
          lr=5e-5, lrschedule='constant', max_grad_norm=0.01, log_interval=30)
    env.close()
