import os
import time
import logging, datetime, csv
import numpy as np
# import os.path as osp
import tensorflow as tf
# from baselines import logger
from collections import deque
# from baselines.common import explained_variance
from utils_OAI import Scheduler, make_path, find_trainable_variables, make_session, set_global_seeds
from utils_OAI import cat_entropy, mse, explained_variance, normalize_obs, discount_with_dones
from utils_OAI import add_to_collection_rnn_state

# TODO nsteps == batch_size
class Model(object):
    def __init__(self, policy, ob_space, ac_space, lr, cliprange,
                 max_grad_norm, units_per_hlayer, activ_fcn, log_interval, logdir,
                 nenvs, nbatch_train, nsteps, ent_coef, vf_coef, keep_model, meta=False):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up PPO learning agent')
        # self.logger.info('nsteps: ' + str(total_timesteps))
        self.num_steps_trained = 0
        self.log_interval = log_interval

        sess = make_session()  # sess = tf.get_default_session()
        nact = ac_space.n
        nbatch = nenvs * nsteps

        self.global_step = tf.get_variable('global_step',
                                           [],
                                           tf.int32,
                                           tf.constant_initializer(0, tf.int32),
                                           trainable=False)

        eval_model = policy(sess, ob_space, ac_space, 1, 1, units_per_hlayer, reuse=False, activ_fcn=activ_fcn)
        act_model = policy(sess, ob_space, ac_space, nenvs, 1, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)
        # train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)
        train_model = policy(sess, ob_space, ac_space, nenvs, nbatch_train//nenvs, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)
        # if meta:
        #     meta_model = policy(sess, ob_space, ac_space, nenvs, nbatch_train//nenvs, units_per_hlayer, reuse=True, activ_fcn=activ_fcn)  # TODO

        A = tf.placeholder(tf.int32, [None], name='A')  # train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None], name='ADV')
        R = tf.placeholder(tf.float32, [None], name='R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        # LR = tf.placeholder(tf.float32, [])
        # CLIPRANGE = tf.placeholder(tf.float32, [])
        LR = lr
        CLIPRANGE = cliprange

        def get_loss(model, placeholder_dict):
            a = placeholder_dict["A"]
            adv = placeholder_dict["ADV"]
            r = placeholder_dict["R"]
            oldneglocpac = placeholder_dict["OLDNEGLOGPAC"]
            oldvpred =placeholder_dict["OLDVPRED"]
            clip_range = placeholder_dict["CLIPRANGE"]

            neglogpac = model.pd.neglogp(a)
            entropy = tf.reduce_mean(cat_entropy(model.pi_logit))  # train_model.pd.entropy())
            vpred = model.vf
            vpredclipped = oldvpred + tf.clip_by_value(model.vf - oldvpred, - clip_range, clip_range)
            vf_losses1 = tf.square(vpred - r)
            vf_losses2 = tf.square(vpredclipped - r)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(oldneglocpac - neglogpac)
            pg_losses = -adv * ratio
            pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - oldneglocpac))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), clip_range)))
            return pg_loss, entropy, vf_loss, vpred, neglogpac, approxkl, clipfrac

        # neglogpac = train_model.pd.neglogp(A)
        # entropy = tf.reduce_mean(cat_entropy(train_model.pi_logit))  # train_model.pd.entropy())
        #
        # vpred = train_model.vf
        # vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # vf_losses1 = tf.square(vpred - R)
        # vf_losses2 = tf.square(vpredclipped - R)
        # vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        # pg_losses = -ADV * ratio
        # pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        # clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        self.input_plchld = {'A':A, 'ADV':ADV, 'R':R, 'OLDNEGLOGPAC':OLDNEGLOGPAC, 'OLDVPRED':OLDVPRED, 'CLIPRANGE':CLIPRANGE}
        pg_loss, entropy, vf_loss, vpred, neglogpac, approxkl, clipfrac = \
            get_loss(train_model, self.input_plchld)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = [trainer.apply_gradients(grads), self.global_step.assign_add(nsteps)]

        if log_interval > 0:
            for g, v in grads:
                if g is not None:
                    tf.summary.histogram("%s-grad" % v.name.replace(':', '_'), g)
            for p in params:
                if p is not None:
                    tf.summary.histogram("train/%s" % p.name.replace(':', '_'), p.value())
            tf.summary.scalar("train/pg_loss", pg_loss)
            tf.summary.scalar("train/vf_loss", vf_loss)
            tf.summary.scalar("train/entropy", entropy)
            tf.summary.histogram("others/ADV", ADV)
            tf.summary.histogram("others/neglocpac", neglogpac)
            tf.summary.histogram("others/vf", vpred)
            self.summary_step = tf.summary.merge_all()

        # lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        # Adding these to collection so we can restore them again
        tf.add_to_collection('inputs', eval_model.X)
        # tf.add_to_collection('pi', eval_model.pi)
        # tf.add_to_collection('pi_logit', eval_model.pi_logit)
        tf.add_to_collection('val', eval_model.vf)
        tf.add_to_collection('step', eval_model.ac)
        if eval_model.initial_state is not None:
            add_to_collection_rnn_state('state_in', eval_model.rnn_state_in)
            add_to_collection_rnn_state('state_out', eval_model.rnn_state_out)

        tf.global_variables_initializer().run(session=sess)

        # def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        def train(obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)  # TODO why?
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, # LR:lr, CLIPRANGE:cliprange,
                    OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.rnn_state_in] = states
                # td_map[train_model.M] = masks  # TODO masks are required to reset rnn state if episode terminated

            policy_loss, value_loss, policy_entropy, app_kl, clipratio, gs = \
                sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, self.global_step, _train], td_map)[:-1]
            # TF summary logging
            if log_interval > 0 and (self.num_steps_trained % self.log_interval == 0):
                self.logger.info('Save summary of network weights, grads and losses.')
                summary_str = sess.run(self.summary_step, td_map)
                self.summary_writer.add_summary(tf.Summary.FromString(summary_str), gs)
            self.num_steps_trained += 1
            return policy_loss, value_loss, policy_entropy, app_kl, clipratio
            #     sess.run(
            #     [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
            #     td_map
            # )[:-1]

        # saver = tf.train.Saver(max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=keep_model)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(f_name):
            gs = sess.run(self.global_step)
            self.logger.info('Save network parameters of model at global step %s' % gs)
            saver.save(sess, os.path.join(logdir, f_name), global_step=gs)
            # ps = sess.run(params)
            # joblib.dump(ps, save_path)

        def load(load_path):
            saver.restore(sess, load_path)
            # loaded_params = joblib.load(load_path)
            # restores = []
            # for p, loaded_p in zip(params, loaded_params):
            #     restores.append(p.assign(loaded_p))
            # sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

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

        self.get_loss = get_loss
        self.trainer = trainer
        self.train = train
        self.train_model = train_model
        self.eval_model = eval_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.test_run = test_run
        # tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

        # Set the summary writer to write to the given logdir if logging is enabled
        if log_interval > 0:
            self.summary_writer = tf.summary.FileWriter(logdir, graph_def=sess.graph_def)
        else:
            self.summary_writer = None

        self.sess = sess

    def get_summary_writer(self):
        return self.summary_writer

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, horizon=100, show_interval=0, summary_writer=None):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.info('Set up %s-step Runner' % nsteps)
        self.env = env
        self.model = model
        nd, = env.observation_space.shape
        self.nenv = nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.reward_window = [deque(maxlen=horizon) for _ in range(nenv)]

        # episode stats - ONLY required in episodic setting
        self.eplength = [0 for _ in range(nenv)]
        self.epreturn = [0 for _ in range(nenv)]
        self.return_threshold = -2.  # 0.0  # 40  # threshold below which no model will be saved.

        # rendering
        self.show_interval = show_interval
        self.ep_idx = [0 for _ in range(nenv)]

        self.summary_writer = summary_writer

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        # epinfos = []
        for _ in range(self.nsteps):
            actions, pi, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            # print(pi)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            obs, rewards, self.dones, _ = self.env.step(actions)
            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)
            self.obs[:] = normalize_obs(obs)
            mb_rewards.append(rewards)

            self.logger.debug('Observations: %s' % self.obs)

            # render only every i-th episode
            if self.show_interval != 0:
                if (self.ep_idx[0] % self.show_interval) == 0:
                    self.env.render()

            self.eplength = [self.eplength[i] + 1 for i in
                             range(self.nenv)]  # Todo use already implemented functions in run_ple_utils!!!
            self.epreturn = [self.epreturn[i] + rewards[i] for i in range(self.nenv)]
            [self.reward_window[i].append(rewards[i]) for i in range(self.nenv)]

            # Check for terminal states in every env - this is only used in terminating version of FlappyBird
            for i, done in enumerate(self.dones):  # i -> environment ID
                if done:
                    self.ep_idx[i] += 1
                    self.obs[i] = self.obs[i] * 0

                    # update tensorboard summary
                    if self.summary_writer is not None:
                        summary = tf.Summary()
                        summary.value.add(tag='envs/environment%s/episode_length' % i,
                                          simple_value=self.eplength[i])
                        summary.value.add(tag='envs/environment%s/episode_reward' % i,
                                          simple_value=self.epreturn[i])
                        self.summary_writer.add_summary(summary, self.ep_idx[i])  # self.global_step.eval())
                        self.summary_writer.flush()
                    # self.retbuffer.append(self.epreturn[i])
                    if self.epreturn[i] > self.return_threshold:
                        self.return_threshold = self.epreturn[i]
                        self.logger.info('Save model at max reward %s' % self.return_threshold)
                        self.model.save('inter_model')
                    self.eplength[i] = 0
                    self.epreturn[i] = 0
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)  # TODO this is an array of tensors, output values instead!!
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]  # 1-step td-error
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        self.logger.debug('Actions: %s' % mb_actions)
        self.logger.debug('Q values: %s' % mb_values)
        # self.logger.debug('Done mask: %s' % mb_masks)  # ?
        self.logger.debug('Observations: %s' % mb_obs)

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, self.reward_window, mb_rewards)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

# nsteps == batchsize
def learn(policy, env, test_env, seed, total_timesteps,
          log_interval, test_interval, show_interval, logdir,
          lr, max_grad_norm, units_per_hlayer, activ_fcn,
          gamma=0.99, vf_coef=0.5, ent_coef=0.01, nsteps=5,
          lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
          early_stop=False, keep_model=2,
          save_model=True, restore_model=False, save_traj=False):

    # if isinstance(lr, float): lr = constfn(lr)
    # else: assert callable(lr)
    # if isinstance(cliprange, float): cliprange = constfn(cliprange)
    # else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    logger = logging.getLogger(__name__)
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches  # TODO number of samples per minibatch in an optimization episode

    make_model = lambda : Model(policy=policy,
                                ob_space=ob_space,
                                ac_space=ac_space,
                                nenvs=nenvs,
                                nbatch_train=nbatch_train,
                                nsteps=nsteps,
                                ent_coef=ent_coef,
                                vf_coef=vf_coef,
                                max_grad_norm=max_grad_norm,
                                activ_fcn=activ_fcn,
                                units_per_hlayer=units_per_hlayer,
                                log_interval=log_interval,
                                logdir=logdir,
                                keep_model=keep_model,
                                lr=lr,
                                cliprange=cliprange)
    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    sum_write = model.get_summary_writer()
    result_path = os.path.join(logdir, 'train_results.csv')
    if save_traj:
        rew_traj = []
        rew_results_path = os.path.join(logdir, ('lr'+str(lr)+'_tracking_results.csv'))
    else:
        rew_results_path = None

    i_sample, i_train = 0, 0
    return_threshold = -2.
    horizon = 100
    avg_rm = deque(maxlen=30)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, horizon=horizon, show_interval=show_interval, summary_writer=sum_write)

    if restore_model:
        for el in os.listdir(logdir):
            if 'final' in el and '.meta' in el:
                # Load pre trained model and set network parameters
                model.load(os.path.join(logdir, el[:-5]))
                # Reset global step parameter.
                model.sess.run(model.global_step.assign(0))

    logger.info('Start Training')
    breaked = False

    # epinfobuf = deque(maxlen=100)
    # tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0  # nbatch should be a multiple of nminibatches
        obs, returns, masks, actions, values, neglogpacs, states, reward_window, rewards = \
            runner.run()  #pylint: disable=E0632  # returns are estimates of the discounted reward

        if rew_results_path is not None:
            rew_traj.append(rewards)


        nbatch_train = nbatch // nminibatches  # number of samples per minibatch
        tstart = time.time()
        # frac = 1.0 - (update - 1.0) / nupdates  # converges to 0
        # lrnow = lr(frac)  #
        # cliprangenow = cliprange(frac)  # cliprange converges to 0

        # Update step
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)  #
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                    mblossvals.append(model.train(*slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches  # minibatch contains batch data from several envs.
            envinds = np.arange(nenvs, dtype=np.int32)
            # print(envinds)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            # print(envsperbatch)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = np.array(envinds[start:end])  # TODO int() does not work here. ensure that indices are integers beforehand
                    # print(mbenvinds)
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    if nenvs == 1:
                        mbstates = states[:]
                    else:
                        # print(states[1])
                        # print(states[0])
                        if type(states) == tuple or type(states) == tf.contrib.rnn.LSTMStateTuple:  # LSTM state
                            mbstates = [el[mbenvinds] for el in states]
                        else:  # GRU state
                            mbstates = states[mbenvinds]
                        # print(mbstates)
                    # mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                    mblossvals.append(model.train(*slices, mbstates))

        if test_interval > 0 and i_train > 0 and (update % test_interval == 0):
            ep_return = model.test_run(test_env, n_eps=10, n_pipes=2000)  # TODO test, whether results.csv is saved properly
            with open(result_path, "a") as csvfile:
                writer = csv.writer(csvfile)
                # ep_return = [str(p) for p in ep_return]
                # ep_return.insert(0, ('step_%s' % i_sample))
                ep_return[0:0] = [i_sample, i_train]
                writer.writerow(ep_return)

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

        # lossvals = np.mean(mblossvals, axis=0)
        # tnow = time.time()
        # fps = int(nbatch / (tnow - tstart))
        # if update % log_interval == 0 or update == 1:
        #     ev = explained_variance(values, returns)
        #     logger.logkv("serial_timesteps", update*nsteps)
        #     logger.logkv("nupdates", update)
        #     logger.logkv("total_timesteps", update*nbatch)
        #     logger.logkv("fps", fps)
        #     logger.logkv("explained_variance", float(ev))
        #     logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        #     logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        #     logger.logkv('time_elapsed', tnow - tfirststart)
        #     for (lossval, lossname) in zip(lossvals, model.loss_names):
        #         logger.logkv(lossname, lossval)
        #     logger.dumpkvs()
        # if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
        #     checkdir = osp.join(logger.get_dir(), 'checkpoints')
        #     os.makedirs(checkdir, exist_ok=True)
        #     savepath = osp.join(checkdir, '%.5i'%update)
        #     print('Saving to', savepath)
        #     model.save(savepath)

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
    # env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

from run_ple_utils import make_ple_envs, make_ple_env  #, arg_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy, LargerLSTMPolicy, GRUPolicy
if __name__ == '__main__':
    seed = 2
    # env = make_ple_envs('FlappyBird-v2', num_env=3, seed=seed,
    #                     trace_length=500, offset=1., amplitude=0.1, fband=[0.0001, 0.005],  # Filtered Random Walk
    #                     nsteps=20, time_interval=[20,200], value_interval=[3,6])            # Random Steps
    # env = make_ple_envs('ContFlappyBird-v1', num_env=1, seed=seed)
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

    logdir = '/home/mara/Desktop/logs/LSTM_PPO'
    logdir = os.path.join(logdir, 'ppo_output1')
    # logdir = os.path.join(logdir, datetime.datetime.today().strftime('%Y_%m_%d_%H%M%S'))

    # learn(LargerMLPPolicy, env,
    learn(policy=LargerLSTMPolicy,
          env=env,
          test_env=test_env,
          seed=seed,
          total_timesteps=10000,  # int(1e7),
          log_interval=0,
          test_interval=0,
          show_interval=0,
          logdir=logdir,
          lr=5e-4,
          # lrschedule='constant',
          max_grad_norm=0.01,
          units_per_hlayer=(24, 24, 24),
          activ_fcn='relu6',
          gamma=0.9,
          vf_coef=0.3,
          ent_coef=0.00005,
          nsteps=64,
          lam=0.95,
          nminibatches=1,
          noptepochs=1,
          cliprange=0.2,
          early_stop=False,
          keep_model=2,
          restore_model=False)
    env.close()
