import os
import os.path as osp
import gym
import time
import joblib
import numpy as np
import tensorflow as tf
import random
import logger
from ple.games.flappybird import FlappyBird
from ple import PLE

from utils_OAI import discount_with_dones
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session
from utils_OAI import cat_entropy, mse
from utils_OAI import process_state

from models_OAI import MlpPolicy, LstmPolicy


class Model(object):

    def __init__(self, policy, ob_shape, ac_space, replay_batch, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = make_session()  # use as many cpu cores as available
        # nact = ac_space.n

        # nbatch = nenvs*nsteps
        if replay_batch is not None:
            nbatch = replay_batch * nsteps
        else:
            nbatch = nsteps

        A = tf.placeholder(tf.int32, [nbatch], name='A')
        ADV = tf.placeholder(tf.float32, [nbatch], name='ADV')
        R = tf.placeholder(tf.float32, [nbatch], name='R')
        LR = tf.placeholder(tf.float32, [], name='LR')

        step_model = policy(sess, ob_shape, ac_space, 1, 1, reuse=False)  # batch = single real sample, nsteps = single step
        train_model = policy(sess, ob_shape, ac_space, nbatch, nsteps, reuse=True)  # batch = depends on replay buffer or not, nsteps = nstep training trajectories

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        # grads = tf.gradients(loss, params)
        # if max_grad_norm is not None:
        #     grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        grads = trainer.compute_gradients(loss)
        _train = trainer.apply_gradients(grads)

        for g, v in grads:
            if g is not None:
                tf.summary.histogram("%s-grad" % v.name, g)
        tf.summary.scalar("pg_loss", pg_loss)
        tf.summary.scalar("vf_loss", vf_loss)
        tf.summary.scalar("entropy", entropy)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, stepidx):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks  # masks those samples which resulted in done episode
            policy_loss, value_loss, policy_entropy, _, ap = sess.run(
                [pg_loss, vf_loss, entropy, _train, train_model.pi],
                td_map
            )
            merge = tf.summary.merge_all()
            summary_str = sess.run(merge, td_map)
            self.summary_writer.add_summary(summary_str, stepidx)
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
        self.summary_writer = tf.summary.FileWriter('/home/mara/Desktop/logs/A2C_OAI', graph_def=sess.graph_def)


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.actionSet = env.getActionSet()
        self.model = model
        nd, = env.getGameStateDims()  # env.observation_space.shape
        nenv = 1  # env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nd)
        # self.obs = np.zeros(nd, dtype=np.float32)  # np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        # self.nc = nc
        env.reset_game()
        self.obs = env.getGameState()

        # params
        self.gamma = gamma  # discount factor
        self.nsteps = nsteps

        # Variables for networks including LSTMs
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]  # TODO only one environment
        assert len(self.dones) == 1, 'dones doesnt fit number of envs: 1'

    def run(self):  # sample n-step minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step([self.obs], self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # obs, rewards, dones, _ = self.env.step(actions)
            a = self.actionSet[int(actions)]
            # print(actions, a)
            rewards = env.act(a)
            dones = env.game_over()
            if dones:
                env.reset_game()
            obs = env.getGameState()
            self.obs = obs

            # Update LSTM states
            self.states = states
            self.dones = [dones]
            # for n, done in enumerate(dones):  # n counts the envs useless statement! It is directly overwritten by the next step without being used!
            #     if done:
            #         self.obs[n] = self.obs[n]*0  # set previous observations to 0

            mb_rewards.append(rewards)

        mb_dones.append(self.dones[0])
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)  # .swapaxes(1, 0)  # .reshape(self.batch_ob_shape)  # dtype was uint8??
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)  # .swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)  # .swapaxes(1, 0) # TODO not needed anymore
        mb_values = np.asarray(mb_values, dtype=np.float32)  # .swapaxes(1, 0)  # TODO not needed anymore
        mb_dones = np.asarray(mb_dones, dtype=np.bool)  # .swapaxes(1, 0)
        mb_masks = mb_dones[:-1]
        mb_dones = mb_dones[1:]
        last_value = self.model.value([self.obs], self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        # for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):  # enumerate not necessay as it is only one env
        #     rewards = rewards.tolist()
        #     dones = dones.tolist()
        #     if dones[-1] == 0:
        #         rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
        #     else:
        #         rewards = discount_with_dones(rewards, dones, self.gamma)
        #     mb_rewards[n] = rewards

        # Compute reward
        rewards = mb_rewards.tolist()
        dones = mb_dones.tolist()
        # print(rewards)
        if dones[-1] == 0:
            rewards = discount_with_dones(rewards+last_value, dones+[0], self.gamma)[:-1]
        else:
            rewards = discount_with_dones(rewards, dones, self.gamma)
        # print(rewards)

        mb_rewards = np.asarray(rewards, dtype=np.float32).squeeze() # reshape([1, -1])
        mb_actions = np.asarray(mb_actions, dtype=np.float32).squeeze() # reshape([1, -1])
        mb_values = np.asarray(mb_values, dtype=np.float32).squeeze()  # reshape([1, -1])
        mb_masks = np.asarray(mb_masks, dtype=np.float32).squeeze()  # reshape([1, -1])
        # mb_rewards = mb_rewards.flatten()
        # mb_actions = mb_actions.flatten()
        # mb_values = mb_values.flatten()
        # mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, nsteps=5, replay_batch=None, total_trainsteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()

    # set global seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.init()
    # ob_space = env.observation_space
    # ac_space = env.action_space
    ac_space = env.getActionSet()  # should be [0,1]
    assert type(ac_space) is list, "ac_space is no list"
    ob_shape = env.getGameStateDims()
    assert type(ob_shape) is tuple, "ob_space is no tuple"

    model = Model(policy=policy, ob_shape=ob_shape, ac_space=ac_space, replay_batch=replay_batch, nsteps=nsteps,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon,
                  total_timesteps=total_trainsteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    if replay_batch is not None:
        nbatch = replay_batch*nsteps
    else:
        nbatch = nsteps

    tstart = time.time()
    for update in range(1, total_trainsteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        # print(actions)
        policy_loss, value_loss, policy_entropy, ap = model.train(obs, states, rewards, masks, actions, values, update)
        # print(ap)
        if update % log_interval == 0 or update == 1:
            # ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            # logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()


if __name__ == '__main__':
    game = FlappyBird(pipe_gap=250)
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    learn(LstmPolicy, env, seed=0, nsteps=40, vf_coef=0.3, ent_coef=0)

