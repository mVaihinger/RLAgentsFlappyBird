import tensorflow as tf
import os, glob
import csv
import numpy as np
import logging
# import datetime

from utils_OAI import set_global_seeds, normalize_obs, get_collection_rnn_state
from run_ple_utils import make_ple_env


def eval_model(render, nepisodes, test_steps, save_traj=False, result_file='test_results.csv', **params):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating learning algorithm...\n')
    logger.info(params["eval_model"])

    logger.debug('\nMake Environment with seed %s' % params["seed"])
    # TODO use different seed for every run!#, allow_early_resets=True)
    # TODO make non-clipped env, even if agent is trained on clipped env
    ple_env = make_ple_env(params["test_env"], seed=params["seed"])

    tf.reset_default_graph()
    set_global_seeds(params["seed"])
    model_idx = []

    if save_traj:
        result_path = os.path.join(params["logdir"], result_file)
    else:
        result_path = None

    recurrent = (params["architecture"] == 'lstm' or params["architecture"] == 'gru')
    if params["eval_model"] == 'final':
        avg_performances = []
        var_performances = []
        maximal_returns = []
        for f in glob.glob(os.path.join(params["logdir"], '*final_model-*.meta')):
            logger.info('Restore model: %s' % f)
            idx = f.find('final_model')
            f_name = f[idx:-5]
            model_idx.append(f_name)
            with tf.Session() as sess:
                OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_dqn_model(sess, logdir=params["logdir"], f_name=f_name,
                                                                     isrnn=recurrent)
                logger.info('Run %s evaluation episodes' % nepisodes)
                model_performance = run_episodes(sess, ple_env, nepisodes, test_steps, render,
                                                 OBS, RNN_S_IN, RNN_S_OUT, PRED_Q, result_path, params["seed"])
                # Add model performance metrics
                avg_performances.append(np.mean(model_performance))
                var_performances.append(np.var(model_performance))
                maximal_returns.append(np.max(model_performance))
            tf.reset_default_graph()

    elif params["eval_model"] == 'inter':
        # Use all stored maximum performance models and the final model.
        # print('Eval now!')
        avg_performances = []
        var_performances = []
        maximal_returns = []
        for f in glob.glob(os.path.join(params["logdir"], '*inter*.meta')):
            logger.info('Restore model: %s' % f)
            idx = f.find('_model')
            f_name = f[idx-5:-5]
            model_idx.append(f_name)
            with tf.Session() as sess:
                OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_dqn_model(sess, logdir=params["logdir"], f_name=f_name,
                                                                     isrnn=recurrent)
                logger.info('Run %s evaluation episodes' % nepisodes)
                model_performance = run_episodes(sess, ple_env, nepisodes, test_steps, render,
                                                 OBS, RNN_S_IN, RNN_S_OUT, PRED_Q, result_path, params["seed"])

                # Add model performance metrics
                avg_performances.append(np.mean(model_performance))
                var_performances.append(np.var(model_performance))
                maximal_returns.append(np.max(model_performance))
            tf.reset_default_graph()
    elif params["eval_model"] == 'analysis':
        # Use all stored maximum performance models and the final model.
        avg_performances = []
        std_performances = []
        maximal_returns = []
        for f in glob.glob(os.path.join(params["logdir"], '*.meta')):
            logger.info('Restore model: %s' % f)
            idx = f.find('_model')
            f_name = f[idx - 5:-5]
            model_idx.append(f_name)
            # print(f_name)
            with tf.Session() as sess:
                OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_dqn_model(sess, logdir=params["logdir"], f_name=f_name,
                                                                     isrnn=recurrent)
                logger.info('Run %s evaluation episodes' % nepisodes)
                model_performance = run_episodes(sess, ple_env, nepisodes, test_steps, render,
                                                 OBS, RNN_S_IN, RNN_S_OUT, PRED_Q, result_path, params["seed"])

                # Add model performance metrics
                avg_performances.append(np.mean(model_performance))
                std_performances.append(np.std(model_performance))
                maximal_returns.append(np.max(model_performance))
            tf.reset_default_graph()
        return model_idx, avg_performances, std_performances

    logger.info(params["logdir"])
    logger.info('Results of the evaluation of the learning algorithm:')
    logger.info('Restored models: %s' % model_idx)
    logger.info('Average performance per model: %s' % avg_performances)
    logger.info('Performance variance per model: %s' % var_performances)
    logger.info('Maximum episode return per model: %s' % maximal_returns)
    ple_env.close()

    if not avg_performances == []:
        return np.mean(avg_performances), np.mean(var_performances), np.mean(maximal_returns)
    else:
        return -3000, 3000, -3000


def restore_dqn_model(sess, logdir, f_name, isrnn):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # g = tf.get_default_graph()  # Shouldn't be set here again, as a new RNG is used without previous seeding.

    # restore the model
    loader = tf.train.import_meta_graph(glob.glob(os.path.join(logdir, (f_name + '.meta')))[0])  #TODO !!!
    # loader = tf.train.import_meta_graph(glob.glob(os.path.join(logdir, 'inter_model-*.meta'))[0])

    # now variables exist, but the values are not initialized yet.
    loader.restore(sess, os.path.join(logdir, f_name))  # restore values of the variables.

    # Load operations from collections
    obs_in = tf.get_collection('inputs')        # network inputs: observations
    predQ_out = tf.get_collection('predQ')      # get predicted Q values
    if isrnn:
        rnn_state_in = get_collection_rnn_state('state_in')  # rnn cell input vector
        rnn_state_out = get_collection_rnn_state('state_out')  # rnn cell output vector
    else:
        rnn_state_in, rnn_state_out = None, None
    return obs_in, rnn_state_in, rnn_state_out, predQ_out


def run_episodes(sess, env, n_eps, n_steps, render, obs_in, rnn_state_in, rnn_state_out, predQ_out, f, seed):
    logger = logging.getLogger(__name__)
    ep_length = []
    ep_return = []
    logger.info('---------------- Episode results -----------------------')
    for i in range(0, n_eps):  # TODO parallelize this here!
        obs = env.reset()
        obs = normalize_obs(obs)
        done = False
        if rnn_state_in is not None:
            if len(rnn_state_in) > 1:
                rnn_s_in = (np.zeros(rnn_state_in[0].shape), np.zeros(rnn_state_in[1].shape))  # init lstm cell vector
            else:
                rnn_s_in = np.zeros(len(rnn_state_in))  # init gru cell vector
        total_return = 0
        total_length = -1
        reward = 0
        i_sample = 0
        if f is not None:
            rew_traj = []

        while not done and (i_sample < n_steps):
            i_sample += 1
            total_length += 1
            total_return += reward  # add reward of previous step, s.t. termination reward is not added anymore.

            if rnn_state_in is not None:
                pQ, rnn_s_out = sess.run([predQ_out, rnn_state_out], feed_dict={obs_in[0]: [obs], rnn_state_in: rnn_s_in})
            else:
                pQ = sess.run([predQ_out], feed_dict={obs_in[0]: [obs]})
            best_ac = np.argmax(pQ)  # greedy policy not epsilon greedy policy
            obs, reward, done, _ = env.step(best_ac)
            # obs, reward, done, _ = env.step(act[0][0])
            obs = normalize_obs(obs)

            if f is not None:
                rew_traj.append(reward)

            if render:
                env.render()

            if rnn_state_in is not None:
                rnn_s_in = rnn_s_out
        logger.info('Episode %s: %s, %s' % (i, total_return, total_length))
        ep_length.append(total_length)
        ep_return.append(total_return)

        if f is not None:
            with open(f, "a") as csvfile:
                writer = csv.writer(csvfile)
                rew_traj[0:0] = [seed, i, np.mean(rew_traj)]
                writer.writerow(rew_traj)

    return ep_return


if __name__ == '__main__':
    logdir = "/home/mara/Desktop/logs/A2C_OAI_NENVS/dqn_output2"

    logger = logging.getLogger()
    ch = logging.StreamHandler()  # Handler which writes to stderr (in red)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(logdir, 'eval.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # Read params from hyperparameter.txt
    params = dict()
    file = open(os.path.join(logdir, 'hyperparams.txt'), 'r')
    for line in file:
        if line is '\n':
            break
        idx = line.find(':')
        p_name = line[:idx]
        p_val = line[idx + 1:]
        try:
            params[p_name] = int(p_val)
        except Exception:
            try:
                params[p_name] = float(p_val)
            except Exception:
                params[p_name] = p_val[1:-1]
    params["eval_model"] = 'all'
    params["logdir"] = logdir
    params["architecture"] = 'lstm'

    # evaluate model
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=4, test_steps=3000, **params)
