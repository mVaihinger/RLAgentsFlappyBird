from run_ple_utils import make_ple_env
from DQN_PLE import DQNAgent
import tensorflow as tf
import os, glob
import numpy as np
import logging

from utils_OAI import set_global_seeds, normalize_obs

def eval_model(render, nepisodes, **params):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating learning algorithm...')

    logger.debug('Make Environment with seed %s' % params["seed"])
    ple_env = make_ple_env(params["env"], seed=params["seed"])  # , allow_early_resets=True)

    tf.reset_default_graph()
    set_global_seeds(params["seed"])
    model_idx = []

    if params["eval_model"] == 'final':
        f = glob.glob(os.path.join(params["logdir"], 'final_model-*.meta'))
        idx = f.find('run')
        f_name = f[idx + 6:-5]
        model_idx.append(f_name)
        with tf.Session() as sess:
            OBS, PRED_Q = restore_dqn_model(sess, logdir=params["logdir"], f_name=f_name)
            avg_performances, var_performances, maximal_returns = \
                run_episodes(sess, ple_env, nepisodes, render, params["epsilon"], OBS, PRED_Q)

        tf.reset_default_graph()

    elif params["eval_model"] == 'all':
        # Use all stored maximum performance models and the final model.
        avg_performances = []
        var_performances = []
        maximal_returns = []
        for f in glob.glob(os.path.join(params["logdir"], '*.meta')):
            logger.info('Restore model: %s' % f)
            idx = f.find('run')
            f_name = f[idx+6:-5]
            model_idx.append(f_name)
            with tf.Session() as sess:
                OBS, PRED_Q = restore_dqn_model(sess, logdir=params["logdir"], f_name=f_name)
                logger.info('Run %s evaluation episodes' % nepisodes)
                avg_model_performance, var_model_performance, maximal_model_return = \
                    run_episodes(sess, ple_env, nepisodes, render, params["epsilon"], OBS, PRED_Q)

                # Add model performance metrics
                avg_performances.append(avg_model_performance)
                var_performances.append(var_model_performance)
                maximal_returns.append(maximal_model_return)
            tf.reset_default_graph()

    logger.info('Results of the evaluation of the learning algorithm:')
    logger.info('Restored models: %s' % model_idx)
    logger.info('Average performance per model: %s' % avg_performances)
    logger.info('Performance variance per model: %s' % var_performances)
    logger.info('Maximum episode return per model: %s' % maximal_returns)
    ple_env.close()

    return np.mean(avg_performances), np.mean(var_performances), np.mean(maximal_returns)

def restore_dqn_model(sess, logdir, f_name):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    g = tf.get_default_graph()

    # restore the model
    loader = tf.train.import_meta_graph(glob.glob(os.path.join(logdir, 'final_model-*.meta'))[0])
    # now variables exist, but the values are not initialized yet.
    loader.restore(sess, os.path.join(logdir, f_name))  # restore values of the variables.

    # Load operations from collections
    obs_in = tf.get_collection('inputs')        # network inputs: observations
    predQ_out = tf.get_collection('predQ')      # get predicted Q values
    return obs_in, predQ_out

def run_episodes(sess, env, n_eps, render, epsilon, obs_in, predQ_out):
    logger = logging.getLogger(__name__)
    ep_length = []
    ep_return = []

    for i in range(0, n_eps):  # TODO parallelize this here!
        obs = env.reset()
        obs = normalize_obs(obs)
        done = False
        total_return = 0
        total_length = 0

        while not done:
            pQ = sess.run([predQ_out], feed_dict={obs_in[0]: [obs]})
            best_ac = np.argmax(pQ)  # greedy policy not, epsilon greedy policy
            obs, reward, done, _ = env.step(best_ac)
            # obs, reward, done, _ = env.step(act[0][0])
            obs = normalize_obs(obs)

            if render:
                env.render()

            total_length += 1
            total_return += reward
        logger.debug('*****************************************')
        logger.debug('Episode %s: %s, %s' % (i, total_return, total_length))
        ep_length.append(total_length)
        ep_return.append(total_return)

    return np.mean(ep_return), np.var(ep_return), np.max(ep_return)


if __name__ == '__main__':
    logdir = "/home/mara/Desktop/logs/DQN/2018-06-12-16-32-47/run1/"

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

    # evaluate model
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=100, **params)
