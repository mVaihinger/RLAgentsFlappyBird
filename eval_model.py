from run_ple_utils import make_ple_envs
from A2C_OAI_NENVS import Model # , LOG_FILE
import tensorflow as tf
import os, glob
import numpy as np

from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy
from utils_OAI import set_global_seeds

def eval_model(render, nepisodes, **params):
    print('Evaluating model...')
    ple_env = make_ple_envs(params["env"], num_env=1, seed=params["seed"], allow_early_resets=True)

    tf.reset_default_graph()
    set_global_seeds(params["seed"])

    nenvs = ple_env.num_envs
    ob_space = ple_env.observation_space
    ac_space = ple_env.action_space
    nd, = ple_env.observation_space.shape

    # Create model: policy, ac_space, ob_space, nenvs and nsteps have to be the same than during training.
    if params["policy"] == 'mlp':
        policy_fn = MlpPolicy
    elif params["policy"] == 'casta':
        policy_fn = CastaPolicy
    elif params["policy"] == 'largemlp':
        policy_fn = LargerMLPPolicy

    model = Model(policy=policy_fn,
                  ob_space=ob_space,
                  ac_space=ac_space,
                  nenvs=nenvs,
                  nsteps=params["nsteps"],
                  vf_coef=params["vf_coeff"],
                  ent_coef=params["ent_coeff"],
                  max_grad_norm=params["max_grad_norm"],
                  lr=params["lr"],
                  lrschedule=params["lrschedule"],
                  units_per_hlayer=(params["units_shared_layer1"],
                                    params["units_shared_layer2"],
                                    params["units_policy_layer"]),
                  total_timesteps=params["total_timesteps"],
                  log_interval=0)

    file = glob.glob(os.path.join(params["logdir"], 'final_model-*.meta'))
    loader = tf.train.import_meta_graph(os.path.join(params["logdir"], file[0]))

    # # Only use final model:
    # with tf.Session() as sess:
    #     loader.restore(sess, tf.train.latest_checkpoint(params["logdir"]))  # load parameter values of last checkpoint into session
    #     obs = ple_env.reset()
    #     done = False
    #     ep_length = []
    #     ep_return = []
    #     total_return = 0
    #     total_length = 0
    #
    #     for i in range(nepisodes):
    #         act, _, _ = model.step(obs)
    #
    #         if done:
    #             ep_length.append(total_length)
    #             ep_return.append(total_return)
    #             break
    #
    #         next_obs, reward, done, _ = ple_env.step(act)
    #         if render:
    #             ple_env.render()
    #
    #         total_length += 1
    #         total_return += reward
    #
    #     ple_env.close()
    #
    #     avg_performance = np.mean(ep_return)
    #     var_performance = np.var(ep_return)
    #     max_return = max(ep_return)

    # Use all stored maximum performance models and the final model.
    avg_performance = []
    var_performance = []
    max_return = []
    for f in glob.glob(os.path.join(params["logdir"], '*.meta')):
        with tf.Session() as sess:
            loader.restore(sess, f[:-5])  # load parameter values of model into session
            obs = ple_env.reset()
            done = False
            ep_length = []
            ep_return = []
            total_return = 0
            total_length = 0

            for _ in range(0, nepisodes):
                while not done:
                    act, _, _ = model.step(obs)
                    next_obs, reward, done, _ = ple_env.step(act)
                    if render:
                        ple_env.render()

                    total_length += 1
                    total_return += reward
                done = False
                total_return = 0
                total_length = 0
                ep_length.append(total_length)
                ep_return.append(total_return)
            # print('aa ' + str(ep_return))
            avg_performance.append(np.mean(ep_return))
            var_performance.append(np.var(ep_return))
            max_return.append(np.max(ep_return))  # TODO why is this here empty??
    ple_env.close()

    # print(avg_performance)
    # print(var_performance)
    # print(max_return)

    return np.mean(avg_performance), np.mean(var_performance), np.mean(max_return)


if __name__ == '__main__':
    LOG_FILE = "/home/mara/Desktop/logs/A2C_OAI_NENVS/2018-05-16 15:31:12.040042"
    f_name = 'inter_model_300.meta'

    seed = 1
    nenvs = 1

    tf.reset_default_graph()
    env = make_ple_envs('FlappyBird-v1', num_env=nenvs, seed=seed)
    set_global_seeds(seed)

    eval_model()

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    # Create model: policy, ac_space, ob_space, nenvs and nsteps have to be the same than during training.
    policy = CastaPolicy
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=50, vf_coef=0.2,
                  ent_coef=1e-7, max_grad_norm=0.01, lr=5e-5, lrschedule='constant', total_timesteps=1000)
    if f_name is None:
        file = glob.glob(os.path.join(LOG_FILE, 'final_model-*.meta'))
    else:
        file = glob.glob(os.path.join(LOG_FILE, f_name))
    loader = tf.train.import_meta_graph(os.path.join(LOG_FILE, file[0]))
    # g = tf.get_default_graph()
    with tf.Session() as sess:
        loader.restore(sess, tf.train.latest_checkpoint(LOG_FILE))  # load parameter values of last checkpoint into session
        obs = env.reset()
        done = False

        # TODO compute average return of each episode

        for i in range(10000):
            # if done:
                # env.reset()

            act, _, _ = model.step(obs)
            print(act)
            # act = sess.run(pi, feed_dict={'Ob:0': obs})
            next_obs, reward, done, _ = env.step(act)
            env.render()

        env.close()

    # env = make_ple_envs('FlappyBird-v1', num_env=nenvs, seed=seed)
    # ac_space = env.action_space
    # ob_space = env.observation_space
    # model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nsteps=5, nenvs=nenvs)
    # env.close()

# Start env

# Load model trained with method A2C, DQN, etc.


# Evaluate the model