from run_ple_utils import make_ple_envs
from A2C_OAI_NENVS import Model # , LOG_FILE
import tensorflow as tf
import os, glob

from models_OAI import CastaPolicy
from utils_OAI import set_global_seeds

LOG_FILE = "/home/mara/Desktop/logs/A2C_OAI_NENVS/2018-05-16 15:31:12.040042"
f_name = 'inter_model_300.meta'
f_name = None


if __name__ == '__main__':
    seed = 1
    nenvs = 1

    tf.reset_default_graph()
    env = make_ple_envs('FlappyBird-v1', num_env=nenvs, seed=seed)
    set_global_seeds(seed)

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