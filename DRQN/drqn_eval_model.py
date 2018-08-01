import tensorflow as tf
import os, glob
import csv
import numpy as np
import logging
import datetime

from utils_OAI import set_global_seeds, normalize_obs, get_collection_rnn_state
from run_ple_utils import make_ple_env


def eval_model(render, nepisodes, **params):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating learning algorithm...')

    logger.debug('Make Environment with seed %s' % params["seed"])
    # TODO make non-clipped env, even if agent is trained on clipped env
    ple_env = make_ple_env(params["env"], seed=params["seed"])  # , allow_early_resets=True)

    tf.reset_default_graph()
    set_global_seeds(params["seed"])
    model_idx = []

    if params["eval_model"] == 'final':
        f = glob.glob(os.path.join(params["logdir"], 'final_model-*.meta'))
        idx = f.find('final_model')
        f_name = f[idx:-5]
        model_idx.append(f_name)
        with tf.Session() as sess:
            OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_drqn_model(sess, logdir=params["logdir"], f_name=f_name)
            model_performance = run_episodes(sess, ple_env, nepisodes, 1000, render,
                                             params["epsilon"], OBS, RNN_S_IN, RNN_S_OUT, PRED_Q)

            # Add model performance metrics
            avg_performances = np.mean(model_performance)
            var_performances = np.var(model_performance)
            maximal_returns = np.max(model_performance)

        tf.reset_default_graph()

    elif params["eval_model"] == 'all':
        # Use all stored maximum performance models and the final model.
        avg_performances = []
        var_performances = []
        maximal_returns = []
        iii = 0
        for f in glob.glob(os.path.join(params["logdir"], '*inter*.meta')):
            logger.info('Restore model: %s' % f)
            idx = f.find('_model')
            f_name = f[idx-5:-5]
            model_idx.append(f_name)
            with tf.Session() as sess:
                OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_drqn_model(sess, logdir=params["logdir"], f_name=f_name)
                logger.info('Run %s evaluation episodes' % nepisodes)
                model_performance = run_episodes(sess, ple_env, nepisodes, 1000, render,
                                                 params["epsilon"], OBS, RNN_S_IN, RNN_S_OUT, PRED_Q)

                # Add model performance metrics
                avg_performances.append(np.mean(model_performance))
                var_performances.append(np.var(model_performance))
                maximal_returns.append(np.max(model_performance))
            tf.reset_default_graph()
    elif params["eval_model"] == "config":
        # Use all stored maximum performance models and the final model.
        avg_performances = []
        var_performances = []
        maximal_returns = []

        # Setup log csv file
        fieldnames = ['model']
        for i in range(nepisodes):
            fieldnames.append(('eps' + str(i)))
        path = os.path.join(params["logdir"], 'results.csv')
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

        # Run evaluation episodes
        models = glob.glob(os.path.join(params["logdir"], '*config_model*.meta'))
        models.sort()
        for f in models:
            logger.info('Restore model: %s' % f)
            idx = f.find('config_model')
            f_name = f[idx:-5]
            model_idx.append(f_name)
            with tf.Session() as sess:
                OBS, RNN_S_IN, RNN_S_OUT, PRED_Q = restore_drqn_model(sess, logdir=params["logdir"], f_name=f_name)
                logger.info('Run %s evaluation episodes' % nepisodes)
                model_performance = run_episodes(sess, ple_env, nepisodes, 2000, render,
                                                 params["epsilon"], OBS, RNN_S_IN, RNN_S_OUT, PRED_Q)  # TODO 1000

                # Add model performance metrics
                avg_performances.append(np.mean(model_performance))
                var_performances.append(np.var(model_performance))
                maximal_returns.append(np.max(model_performance))
            tf.reset_default_graph()

            # Save episode information in csv file for further analysis.
            # Each row contains nepisodes episodes using the current model "f_name".
            with open(path, "a") as csvfile:  # TODO add real returns
                writer = csv.writer(csvfile)
                model_performance = [str(p) for p in model_performance]
                model_performance.insert(0, f_name)
                writer.writerow(model_performance)

    logger.info('Results of the evaluation of the learning algorithm:')
    logger.info('Restored models: %s' % model_idx)
    logger.info('Average performance per model: %s' % avg_performances)
    logger.info('Performance variance per model: %s' % var_performances)
    logger.info('Maximum episode return per model: %s' % maximal_returns)
    ple_env.close()

    if len(avg_performances) > 0:
        return np.mean(avg_performances), np.mean(var_performances), np.mean(maximal_returns)
    else:
        return -5, 0, -5


def restore_drqn_model(sess, logdir, f_name):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    g = tf.get_default_graph()

    # restore the model
    loader = tf.train.import_meta_graph(glob.glob(os.path.join(logdir, 'final_model-*.meta'))[0])
    # now variables exist, but the values are not initialized yet.
    loader.restore(sess, os.path.join(logdir, f_name))  # restore values of the variables.

    # Load operations from collections
    obs_in = tf.get_collection('inputs')            # network inputs: observations
    rnn_state_in = get_collection_rnn_state('state_in')    # rnn cell input vector
    rnn_state_out = get_collection_rnn_state('state_out')  # rnn cell output vector
    predQ_out = tf.get_collection('predQ')          # get predicted Q values
    return obs_in, rnn_state_in, rnn_state_out, predQ_out


def run_episodes(sess, env, n_eps, n_pipes, render, epsilon, obs_in, rnn_state_in, rnn_state_out, predQ_out):
    logger = logging.getLogger(__name__)
    ep_length = []
    ep_return = []
    logger.info('---------------- Episode results -----------------------')
    for i in range(0, n_eps):  # TODO parallelize this here!
        obs = env.reset()
        obs = normalize_obs(obs)
        done = False
        rnn_s_in = (np.zeros(rnn_state_in[0].shape), np.zeros(rnn_state_in[1].shape))  # init rnn cell vector
        total_return = 0
        total_length = 0

        while not done and (total_return < n_pipes):
            pQ, rnn_s_out = sess.run([predQ_out, rnn_state_out], feed_dict={obs_in[0]: [obs], rnn_state_in: rnn_s_in})
            best_ac = np.argmax(pQ)  # greedy policy not epsilon greedy policy
            obs, reward, done, _ = env.step(best_ac)
            # obs, reward, done, _ = env.step(act[0][0])
            obs = normalize_obs(obs)

            if render:
                env.render()

            total_length += 1
            total_return += reward
            rnn_s_in = rnn_s_out
        logger.info('Episode %s: %s, %s' % (i, total_return, total_length))
        ep_length.append(total_length)
        ep_return.append(total_return)

    return ep_return


if __name__ == '__main__':
    logdir = "/home/mara/Desktop/logs/DQN/dqn_rnn_output"

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
    params["eval_model"] = 'config'
    params["logdir"] = logdir

    # evaluate model
    avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=4, **params)
