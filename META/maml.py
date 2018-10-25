import os
import csv
import numpy as np
import tensorflow as tf
import A2C.A2C_OAI_NENVS, PPO.ppo
from utils_OAI import set_global_seeds
from utils_OAI import find_trainable_variables
import logging
from PPO.ppo import Runner

# TODO use different learning rates for fast and meta updates. fast updates can have a rather high lr.


def get_meta_model_cls(base_agent_cls):
    class MAML(base_agent_cls):
        def __init__(self, base_agent, meta_batchsz, meta_task_steps, **kwargs):
            super(MAML, self).__init__(meta=True, **kwargs)
            self.base_agent = base_agent
            self.train_vars = tf.trainable_variables("model")

            # add meta_model that gets a batch of shape (meta_batchsz, K)
            policy = kwargs["policy"]
            self.m_model = policy(self.sess, kwargs["ob_space"], kwargs["ac_space"], meta_batchsz, meta_task_steps,
                                  kwargs["units_per_hlayer"], reuse=True, activ_fcn=kwargs["activ_fcn"])

            # add meta losses w.r.t to meta_model predictions
            m_pg_loss, m_entropy, m_vf_loss, m_vpred, m_neglogpac, m_approxkl, m_clipfrac = \
                self.get_loss(self.m_model, self.input_plchld)
            m_loss = m_pg_loss + m_entropy * kwargs["ent_coef"] + m_vf_loss * kwargs["vf_coef"]

            m_params = find_trainable_variables("model")
            # m_trainer = tf.train.AdamOptimizer(learning_rate=kwargs["lr"])
            m_gradients = self.trainer.compute_gradients(m_loss)
            m_grads, m_variables = zip(*m_gradients)
            if kwargs["max_grad_norm"] is not None:
                m_grads, m_grad_norm = tf.clip_by_global_norm(m_grads, kwargs["max_grad_norm"])
            m_grads = list(zip(m_grads, m_params))
            _m_train = [self.trainer.apply_gradients(m_grads),
                        self.global_step.assign_add(meta_batchsz*meta_task_steps)]

            tf.global_variables_initializer().run(session=self.sess)

            if self.base_agent == 'ppo':
                def mtrain(obs, returns, actions, values, neglogpacs, states=None):
                    advs = returns - values
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)  # TODO why?
                    td_map = {self.m_model.X: obs,
                              self.input_plchld["A"]: actions,
                              self.input_plchld["ADV"]: advs,
                              self.input_plchld["R"]: returns,
                              self.input_plchld["OLDNEGLOGPAC"]: neglogpacs,
                              self.input_plchld["OLDVPRED"]: values}
                    if states is not None:
                        td_map[self.m_model.rnn_state_in] = states
                    policy_loss, value_loss, policy_entropy, app_kl, clipratio, gs = self.sess.run(
                        [m_pg_loss, m_vf_loss, m_entropy, m_approxkl, m_clipfrac, self.global_step, _m_train],
                        td_map)[:-1]
                    self.num_steps_trained += 1
                    return policy_loss, value_loss, policy_entropy, app_kl, clipratio

            elif self.base_agent == 'a2c':
                def mtrain(obs, states, rewards, actions, values):
                    advs = rewards - values  # Estimate for A = Q(s,a) - V(s)
                    td_map = {self.m_model.X: obs,
                              self.input_plchld["A"]: actions,
                              self.input_plchld["ADV"]: advs,
                              self.input_plchld["R"]: rewards}
                    if states is not None:
                        td_map[self.m_model.rnn_state_in] = states
                    policy_loss, value_loss, policy_entropy, test, ap, global_step = self.sess.run(
                        [m_pg_loss, m_vf_loss, m_entropy, _m_train, self.m_model.pi, self.global_step], td_map)
                    self.num_steps_trained += 1
                    return policy_loss, value_loss, policy_entropy, ap

            self.mtrain = mtrain

        def meta_train(self, tb_obs, tb_states, tb_rewards, tb_actions, tb_values, tb_dones=None, tb_neglogpacs=None):
            if self.base_agent == 'ppo':
                return self.mtrain(tb_obs, tb_rewards, tb_actions, tb_values, tb_neglogpacs, tb_states)
            elif self.base_agent == 'a2c':
                return self.mtrain(tb_obs, tb_states, tb_rewards, tb_actions, tb_values)

        def fast_train(self, tb_obs, tb_states, tb_rewards, tb_actions, tb_values, tb_dones=None, tb_neglogpacs=None):
            if self.base_agent == 'ppo':
                return self.train(tb_obs, tb_rewards, tb_dones, tb_actions, tb_values, tb_neglogpacs, tb_states)
            elif self.base_agent == 'a2c':
                return self.train(tb_obs, tb_states, tb_rewards, tb_actions, tb_values)

        def meta_test(self, tb_obs, tb_states, tb_rewards, tb_actions, tb_values, lr, cliprange, tb_dones, tb_neglogpacs):
            # TODO add evaluation model that updates parameter every K samples
            pass

        def get_trainable_vars_vals(self):
            train_vars_values = [p.eval(session=self.sess) for p in tf.trainable_variables()]
            return train_vars_values

        def set_trainable_vars_vals(self, train_vars_values):
            for var, val in zip(self.train_vars, train_vars_values):
                self.sess.run(var.assign(val))

    return MAML


def meta_learn(base_agent, policy, env, test_env, seed, total_timesteps,
               log_interval, test_interval, show_interval, logdir, keep_model,
               lr, max_grad_norm, units_per_hlayer, activ_fcn,
               gamma=0.99, lam=0.95, vf_coeff=0.5, ent_coeff=0.01,

               K=20, train_batchsz=1, kshot=2, test_batchsz=1, meta_batchsz=1,
               test_stage=False, **kwargs):
                # nbatch_train=4, cliprange=0.2,# PPO variables

    logger = logging.getLogger(__name__)
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    nd, = env.observation_space.shape
    ob_space = env.observation_space
    ac_space = env.action_space

    # Init model args
    model_args = dict()
    for k, v in [['policy', policy], ['ob_space', ob_space], ['ac_space', ac_space], ['max_grad_norm', max_grad_norm],
                 ['units_per_hlayer', units_per_hlayer], ['activ_fcn', activ_fcn], ['log_interval', log_interval],
                 ['logdir', logdir], ['nenvs', nenvs], ['ent_coef', ent_coeff], ['vf_coef', vf_coeff],
                 ['keep_model', keep_model], ['base_agent', base_agent], ['lr', lr]]:
        model_args[k] = v

    # Add base agent specific parameters to model_args
    if base_agent == 'ppo':
        base_agent_cls = PPO.ppo.Model
        model_args["nbatch_train"] = K * nenvs
        model_args["nsteps"] = K
        model_args["cliprange"] = kwargs["cliprange"]
    elif base_agent == 'a2c':
        base_agent_cls = A2C.A2C_OAI_NENVS.Model
        model_args["batch_size"] = K
        model_args["lr"] = lr
    else:
        raise Exception('Base Agent %s is not implemented yet' % base_agent)

    # Init meta model
    META_MODEL = get_meta_model_cls(base_agent_cls)
    model = META_MODEL(meta_batchsz=meta_batchsz*test_batchsz, meta_task_steps=K, **model_args)
    sum_write = model.get_summary_writer()
    result_path = os.path.join(logdir, 'meta_train_results.csv')

    # Init worker, which includes data processing, i.e. discounting
    runner = Runner(env=env, model=model, nsteps=K, gamma=gamma, lam=lam, horizon=100,
                    show_interval=show_interval, summary_writer=sum_write)

    i_kshot_training = 0
    i_sample = 0

    if not test_stage:
        steps_per_meta_update = (((train_batchsz * K) * kshot) + (test_batchsz * K) * 1) * meta_batchsz
        for meta_update in range(total_timesteps // steps_per_meta_update):
            print('meta update %s' % meta_update)
            test_obs, test_rewards, test_actions, test_values, test_dones, test_neglogpacs, test_states = \
                [], [], [], [], [], [], []  # init train batches

            init_meta_param = model.get_trainable_vars_vals()

            for t in range(meta_batchsz):
                i_kshot_training += 1
                # Run <Kshot> Fast Training Updates of model parameters
                runner.nsteps = train_batchsz * K
                for k in range(kshot):
                    i_sample += runner.nsteps
                    # Sample <train_batchsz> trajectories with length <K>
                    # and process samples, i.e. discounting and advantage estimation
                    tb_obs, tb_returns, tb_dones, tb_actions, tb_values, tb_neglogpacs, tb_states, reward_window, _ = \
                        runner.run()
                    model.fast_train(tb_obs, tb_states, tb_returns, tb_actions, tb_values,
                                     tb_dones=tb_dones, tb_neglogpacs=tb_neglogpacs)
                    print(model.sess.run([model.train_model.pi], {model.train_model.X:tb_obs})) #, model.train_model.rnn_state_in: tb_states}))

                if test_interval > 0 and (i_kshot_training % test_interval == 0):
                    ep_return = model.test_run(test_env, n_eps=10, n_pipes=2000)
                    with open(result_path, "a") as csvfile:
                        writer = csv.writer(csvfile)
                        # ep_return = [str(p) for p in ep_return]
                        # ep_return.insert(0, ('step_%s' % i_sample))
                        ep_return[0:0] = [i_sample, i_kshot_training]
                        writer.writerow(ep_return)

                # Test Performance of kshot model on <test_batchsz> K-length trajectories.
                runner.nsteps = K
                for i in range(test_batchsz):
                    i_sample += runner.nsteps
                    # Sample trajectory and process samples, i.e. discounting and advantage estimation
                    obs, returns, dones, actions, values, neglogpacs, states, reward_window, _ = runner.run()

                    # Add recent experience to train batches fpr fast updates
                    test_obs.append(obs)
                    test_rewards.append(returns)
                    test_actions.append(actions)
                    test_values.append(values)
                    test_dones.append(dones)
                    test_neglogpacs.append(neglogpacs)
                    test_states.append(states)

                # Reset model to initial param values before fast updates
                model.set_trainable_vars_vals(init_meta_param)

            # Train meta model on test error, based on test samples and estimates with initial parameter vector.
            # Reshape test_samples:
            test_obs = np.asarray(test_obs).swapaxes(0, 1).reshape(-1, nd)
            test_rewards = np.asarray(test_rewards).swapaxes(0, 1).flatten()  # Should be K*test_batchsz,1
            test_actions = np.asarray(test_actions).swapaxes(0, 1).flatten()
            test_values = np.asarray(test_values).swapaxes(0, 1).flatten()
            test_dones = np.asarray(test_dones).swapaxes(0, 1).flatten()
            test_neglogpacs = np.asarray(test_neglogpacs).swapaxes(0, 1).flatten()
            test_states = np.asarray(test_states).swapaxes(0, 1).squeeze()
            test_states = tuple(test_states)
            model.meta_train(test_obs, test_states, test_rewards, test_actions, test_values,
                             tb_dones=test_dones, tb_neglogpacs=test_neglogpacs)

    else:
        # kshot learning toadapt to environment.
        # how often is meta policy updated
        # Is parameter reset to original meta policy after every kshot sequence??
        #
        pass

# a = get_meta_model(PPO.ppo.Model(policy=GRUPolicy, ob_space=1, ac_space=1, max_grad_norm=0.1,
        # units_per_hlayer=(2,2,2), activ_fcn='relu6', log_interval=5, logdir='lalala', nenvs=2, nbatch_train=1,
        # nsteps=20, ent_coef=0.000044, vf_coef=0.11, keep_model=2), 'ppo')
