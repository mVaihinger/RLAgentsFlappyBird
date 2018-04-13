# Based on Advantage Actor-Critic algorithm by Mnih et al. (2016)
# https://arxiv.org/pdf/1602.01783.pdf

# Learn a policy which maximizes the advantage of actions in a trajectory. The critic learns the state-value
# function which is used to estimate the advantage (return - value). The target for the critic is the
# n-step TD-error. On-policy method. No mini-batching used

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import logger
from models import SharedMLP, LSTM_CatPD, LSTM_SM
from utils import LrDecay
from utils import cat_entropy, discount_with_dones, process_state
from utils import eval_model


class ActorCritic:
    def __init__(self, state_dim, n_actions, n_steps, vf_coeff=0.5, entropy_coeff=0.001, lr=0.0007, lr_decay=0.99,
                 fuzz_factor=0.00005, total_timesteps=800000,  max_grad_norm=0.5, scope='actor_critic'):
                # fuzz_factor was called epsilon
        sess = tf.Session()  # TODO add CPU config information

        # Targets in loss computation
        advantage = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')  # advantage of the chosen action
        discounted_reward = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')  # value function target
        action = tf.placeholder(dtype=tf.int32, shape=[None], name='action_in')  # action index
        LR = tf.placeholder(dtype=tf.float32, shape=[])  # learning rate

        # target_model = SharedMLP(sess, state_dim, n_actions)  # used to predict action probs and state values
        # train_model = SharedMLP(sess, state_dim, n_actions)  #, reuse=True)
        # target_model = LSTM(sess, state_dim, n_actions, n_steps=n_steps)
        # used to predict action probs and state values
        train_model = LSTM_SM(sess, state_dim, n_actions, n_steps=n_steps)

        action_onehot = tf.one_hot(action, n_actions, dtype=tf.float32)
        chosen_action_prob = tf.reduce_sum(train_model.ap_out * action_onehot, 1)



        # Compute losses: policy gradient loss (= advantage of the action), the value function loss
        # (Mean Squared error of 1-step TD-target) and an additional regulator regularizing the entropy of the policy,
        # to enhance exploration
        # vf_loss = mean(discounted_reward - estimated_value)²
        # pg_loss = mean(log(action_probs) * advantages)

        pg_loss = - tf.reduce_sum(tf.log(chosen_action_prob)*advantage)
        # action_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.ap_out, labels=action)
        # pg_loss = tf.reduce_mean(action_log_prob * advantage)

        vf_loss = tf.reduce_mean(tf.squared_difference(train_model.vf_out, discounted_reward) / 2.)
        entropy = tf.reduce_mean(cat_entropy(train_model.ap_out))
        loss = pg_loss + vf_coeff * vf_loss - entropy_coeff * entropy

        # Compute gradient of the expected reward w.r.t. the policy parameters
        with tf.variable_scope("model"):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        # clip gradients eventually
        if max_grad_norm is not None:
            # correct way of clipping but slower than clip_by_norm
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = zip(grads, params)
        # grads = list(zip(grads, params))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=lr_decay, epsilon=fuzz_factor)
        train_step = optimizer.apply_gradients(grads)

        _lr = LrDecay(v_init=lr, decay=lr_decay, n_step=total_timesteps)

        # training of training model
        def trainActorCritic(obs, actions, discounted_rewards, values, dones, states):
            adv = discounted_rewards - values
            # enter advantage as one-hot vector
            # adv = [tf.one_hot(a, 1)*adv for a in actions]
            for i in range(len(obs)):
                lr_cur = _lr.value()
            if states is not None:  # LSTM network
                train_dict = {train_model.obs_in: obs,
                              train_model.D: dones,
                              train_model.LS: states,
                              advantage: adv,
                              action: actions,
                              discounted_reward: discounted_rewards,
                              LR: lr_cur}
            else:  # MLP network
                train_dict = {train_model.obs_in: obs,
                              advantage: adv,
                              action: actions,
                              discounted_reward: discounted_rewards,
                              LR: lr_cur}
            # policy_loss, value_loss, policy_entropy, _, ap, a = sess.run([pg_loss,
            #                                                               vf_loss,
            #                                                               entropy,
            #                                                               train_step,
            #                                                               train_model.ap_out,
            #                                                               train_model.a0],
            #                                                              train_dict)
            policy_loss, value_loss, policy_entropy, _, aprob = sess.run([pg_loss,
                                                                          vf_loss,
                                                                          entropy,
                                                                          train_step,
                                                                          train_model.ap_out],
                                                                         train_dict)
            return policy_loss, value_loss, policy_entropy, aprob

        # def save_params():
        #
        # def load_params():

        self.train = trainActorCritic
        self.train_model = train_model
        # self.target_model = target_model
        # self.step = target_model.step
        # self.value = target_model.value
        # self.initial_states = target_model.initial_states
        self.target_model = train_model
        self.step = train_model.step
        self.value = train_model.value
        self.initial_states = train_model.initial_states
        # self.save = save_params
        # self.load = load_params
        tf.global_variables_initializer().run(session=sess)


class Simulation:
    def __init__(self, env, model, nsteps=5, discount=0.99):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.discount = discount
        self.done = 0
        self.states = model.initial_states
        self.obs = None
        self.action_set = self.env.getActionSet()
        self.n_actions = len(self.action_set)
        self.total_return = 0
        print(self.action_set)

    def start_episode(self):
        env.reset_game()
        self.obs = list(env.getGameState())
        self.done = 0
        self.total_return = 0

    def run_nsteps(self):
        b_observations, b_rewards, b_actions, b_values, b_dones = [], [], [], [], []
        b_states = self.states
        for m in range(self.nsteps):
            time.sleep(0.001)
            # action, ap, value, states, _ = self.model.step([self.obs], [self.done], self.states)
            # action = np.random.choice(np.arange(self.n_actions), p=ap[0])
            # reward = env.act(self.action_set[action])
            a_dist, value, states = self.model.step([self.obs], [self.done], self.states)
            action = np.random.choice(np.arange(self.n_actions), p=a_dist[0])
            reward = env.act(self.action_set[action])
            reward += 0.1
            # print('%s %s -- %s' % (ap[0], action, reward)
            if env.game_over():
                done = 1
                obs = list(env.getGameState())
                if abs(self.obs[0] - self.obs[3]) <= 24 or abs(self.obs[0] - self.obs[4]) <= 24:
                    reward = -3. # penalize it less if agent hits tunnel edges
            else:
                done = 0
                obs = list(env.getGameState())
            if reward == 1.:
                print(reward, self.obs)
                print(reward, obs)
            self.states = states
            self.done = done
            b_dones.append(done)
            b_observations.append(np.copy(self.obs))
            b_actions.append(action)
            b_rewards.append(reward)
            b_values.append(value)

            self.obs = obs  # obs = next_obs
            if done:
                break

        self.total_return += reward

        # convert lists to numpy arrays and flatten arrays
        b_observations = np.asarray(b_observations, dtype=np.float32)
        b_rewards = np.asarray(b_rewards, dtype=np.int8).flatten()
        b_actions = np.asarray(b_actions, dtype=np.uint8).flatten()
        b_values = np.asarray(b_values, dtype=np.float32).flatten()
        b_dones = np.asarray(b_dones, dtype=np.int8).flatten()
        next_value = self.model.value([self.obs], [self.done], self.states)
        if b_dones[-1] == 0:
            discounted_rewards = discount_with_dones(b_rewards.tolist()+next_value.tolist(), b_dones.tolist()+[0], self.discount)[:-1]
        else:
            discounted_rewards = discount_with_dones(b_rewards, b_dones, self.discount)
        return b_observations, discounted_rewards, b_actions, b_values, b_dones, b_states


def learn(env, sess, seed, nsteps=5, total_timesteps=int(80e4), discount=0.5, entropy_coeff=0.01, lr=7e-4, lr_decay=0.99,
          fuzz_factor=0.00001,  max_grad_norm=0.5, log_interval=100):
    env.init()
    action_set = env.getActionSet()
    n_actions = len(action_set)
    state_dim = env.getGameState().size    # Reset environment

    total_returns = []

    # Init actorCritic
    actor_critic = ActorCritic(state_dim, n_actions, nsteps, discount, entropy_coeff, lr, lr_decay, fuzz_factor,
                               total_timesteps, max_grad_norm)
    sim = Simulation(env, actor_critic, nsteps=nsteps, discount=discount)
    sim.start_episode()
    e_cnt = 0
    for nupdate in range(int(total_timesteps/nsteps)):
        if env.game_over():
            # done = True
            total_returns.append(sim.total_return)
            sim.start_episode()
            e_cnt = e_cnt+1

        # Collect n-step trajectories
        obs, rewards, actions, values, dones, states = sim.run_nsteps()

        # Update train_model
        policy_loss, value_loss, policy_entropy, a_dist = \
            actor_critic.train(obs, actions, rewards, values, dones, states)
        # print('action probs:')
        # print(ap[0], a)

        if nupdate % log_interval == 0 or nupdate == 1:
            # ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", nupdate)
            logger.record_tabular("nepisode", e_cnt)
            # logger.record_tabular("total_timesteps", nupdate * nsteps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("avg. total return", np.mean(total_returns[-(min(len(total_returns), 100)):]))
            # logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    return actor_critic


if __name__ == "__main__":
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)

    sess = tf.Session()
    seed = 1
    # Runt training
    model = learn(env, sess, seed, max_grad_norm=5, nsteps=4)

    plt.figure()
    plt.plot([1, 2, 3])
    plt.show()

    # Evaluation of trained model
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    n_eps = 500
    rewards = eval_model(env, model, n_eps)

    # Reward per episode
    fig = plt.figure()
    plt.title('Rewards per episodes')
    xscale = range(0, n_eps)
    plt.plot(xscale, rewards, label='AC')
    plt.legend()
    plt.ylabel('reward')
    plt.xlabel('episode')

    # Average reward of last N eps
    fig = plt.figure()
    N = 100  # moving average window
    plt.title('Average Rewards (MAW = %s eps)' % N)
    xscale = range(0, n_eps - N + 1)
    plt.plot(xscale, np.convolve(rewards, np.ones((N,)) / N, mode='valid'), label='AC')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('reward')

    print('Average total return:')
    print('AC: %s' % np.mean(rewards))

    plt.show()
