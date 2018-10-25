#!/usr/local/bin/python3.6
#  parse args:
# --method $MTHD --logdir $LOGDIR --seed $SEED --test_env $TEST_ENV --total_timesteps $TOTAL_TESTSTEPS"
#

# Use the same agent configuration as during training. - see run_remote_Training.sh for configurations.

import os, sys
import argparse, logging

print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
import PPO.ppo, A2C.A2C_OAI_NENVS, DQN.DQN_PLE
from models_OAI import DQN_smac,LSTM_DQN, GRU_DQN, LargerMLPPolicy, LargerLSTMPolicy, GRUPolicy
from run_ple_utils import make_ple_envs, make_ple_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='the learning rate of the tracking agent', type=float, default=0.05)
    parser.add_argument('--method', help='The base method of the agent', type=str, default='LSTM_DQN')
    parser.add_argument('--logdir', help='The directory where final models are located and results are stored', type=str,
                        default='/home/mara/Desktop/logs/LSTM_DQN_test/dqn_output2')
    # parser.add_argument('--seed', help='Random Seed of the test environment. Nonstationarities are pregenerated, hence seeds 100-119 are possible.', type=float, default=100)
    parser.add_argument('--test_env', help='Type of test environment, describing the respective experiment.', type=str, default='ContFlappyBird-hNS-nrf0-train-v0')
    parser.add_argument('--total_timesteps', help='Number of interactions in the test sequence.', type=int, default=5e2)
    parser.add_argument('--restore_model', help='shall a pre defined model be used?', type=bool, default=True)
    args = parser.parse_args()
    print(args)

    exp_dir = args.logdir

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(exp_dir, ('lr' + str(args.lr) + '_track.log')))  # create file first
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if 'output' in exp_dir and not 'msub' in exp_dir:
        logger.info('logdir: '+ exp_dir)

        max_grad_norm = 0.01
        nsteps = 64

        for seed in range(100, 120):
            env = make_ple_envs(args.test_env, num_env=1, seed=seed)

            # switch case between all possible models. Add chosen configs here:

            # TODO implement that reward trajectory is stored during tracking.
            if 'LSTM_PPO' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = PPO.ppo.learn(LargerLSTMPolicy,
                                  env=env,
                                  test_env=None,
                                  seed=seed,
                                  total_timesteps=args.total_timesteps,
                                  log_interval=0,
                                  test_interval=0,
                                  show_interval=0,
                                  logdir=exp_dir,
                                  lr=args.lr,
                                  # lrschedule=args.lrschedule,
                                  max_grad_norm=max_grad_norm,
                                  units_per_hlayer=(24,24,24),
                                  activ_fcn='relu6',
                                  gamma=0.9,
                                  vf_coef=0.3,
                                  ent_coef=0.00005,
                                  nsteps=nsteps,
                                  nminibatches=1,
                                  noptepochs=1,
                                  restore_model=args.restore_model,
                                  save_model=False,
                                  save_traj=True)
            elif 'GRU_PPO' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = PPO.ppo.learn(GRUPolicy,
                                  env=env,
                                  test_env=None,
                                  seed=seed,
                                  total_timesteps=args.total_timesteps,
                                  log_interval=0,
                                  test_interval=0,
                                  show_interval=0,
                                  logdir=exp_dir,
                                  lr=args.lr,
                                  # lrschedule=args.lrschedule,
                                  max_grad_norm=max_grad_norm,
                                  units_per_hlayer=(24, 24, 24),
                                  activ_fcn='relu6',
                                  gamma=0.9,
                                  vf_coef=0.1,
                                  ent_coef=0.00001,
                                  nsteps=nsteps,
                                  nminibatches=1,
                                  noptepochs=1,
                                  restore_model=args.restore_model,
                                  save_model=False,
                                  save_traj=True)
            elif 'PPO' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = PPO.ppo.learn(LargerMLPPolicy,
                                  env=env,
                                  test_env=None,
                                  seed=seed,
                                  total_timesteps=args.total_timesteps,
                                  log_interval=0,
                                  test_interval=0,
                                  show_interval=0,
                                  logdir=exp_dir,
                                  lr=args.lr,
                                  # lrschedule=args.lrschedule,
                                  max_grad_norm=max_grad_norm,
                                  units_per_hlayer=(28, 59, 21),
                                  activ_fcn='mixed',
                                  gamma=0.88,
                                  vf_coef=0.21,
                                  ent_coef=0.00007,
                                  nsteps=128,
                                  nminibatches=2,
                                  noptepochs=4,
                                  restore_model=args.restore_model,
                                  save_model=False,
                                  save_traj=True)
            elif 'LSTM_A2C' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = A2C.A2C_OAI_NENVS.learn(LargerLSTMPolicy,
                                            env=env,
                                            test_env=None,
                                            seed=seed,
                                            total_timesteps=args.total_timesteps,
                                            log_interval=0,
                                            test_interval=0,
                                            show_interval=0,
                                            logdir=exp_dir,
                                            lr=args.lr,
                                            # lrschedule=args.lrschedule,
                                            max_grad_norm=max_grad_norm,
                                            units_per_hlayer=(64, 22, 47),
                                            activ_fcn='mixed',
                                            gamma=0.64,
                                            vf_coef=0.01,
                                            ent_coef=0.00007,
                                            batch_size=nsteps,
                                            restore_model=args.restore_model,
                                            save_model=False,
                                            save_traj=True)
            elif 'GRU_A2C' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = A2C.A2C_OAI_NENVS.learn(GRUPolicy,
                                            env=env,
                                            test_env=None,
                                            seed=seed,
                                            total_timesteps=args.total_timesteps,
                                            log_interval=0,
                                            test_interval=0,
                                            show_interval=0,
                                            logdir=exp_dir,
                                            lr=args.lr,
                                            # lrschedule=args.lrschedule,
                                            max_grad_norm=max_grad_norm,
                                            units_per_hlayer=(14, 35, 79),
                                            activ_fcn='elu',
                                            gamma=0.8,
                                            vf_coef=0.2,
                                            ent_coef=0.00002,
                                            batch_size=nsteps,
                                            restore_model=args.restore_model,
                                            save_model=False,
                                            save_traj=True)
            elif 'A2C' == args.method:
                env = make_ple_envs(args.test_env, num_env=1, seed=seed)
                _ = A2C.A2C_OAI_NENVS.learn(LargerMLPPolicy,
                                            env=env,
                                            test_env=None,
                                            seed=seed,
                                            total_timesteps=args.total_timesteps,
                                            log_interval=0,
                                            test_interval=0,
                                            show_interval=0,
                                            logdir=exp_dir,
                                            lr=args.lr,
                                            # lrschedule=args.lrschedule,
                                            max_grad_norm=max_grad_norm,
                                            units_per_hlayer=(78, 35, 17),
                                            activ_fcn='mixed',
                                            gamma=0.94,
                                            vf_coef=0.36,
                                            ent_coef=0.000036,
                                            batch_size=nsteps,
                                            restore_model=args.restore_model,
                                            save_model=False,
                                            save_traj=True)
            elif 'LSTM_DQN' == args.method:
                env = make_ple_env(args.test_env, seed=seed)
                _ = DQN.DQN_PLE.q_learning(q_network=LSTM_DQN,
                                           env=env,
                                           test_env=None,
                                           seed=seed,
                                           total_timesteps=args.total_timesteps,
                                           log_interval=0,
                                           test_interval=0,
                                           show_interval=0,
                                           logdir=exp_dir,
                                           lr=args.lr,
                                           max_grad_norm=max_grad_norm,
                                           units_per_hlayer=(71, 65, 85),
                                           activ_fcn='relu6',
                                           gamma=0.85,
                                           epsilon=0.22,
                                           epsilon_decay=0.975,
                                           buffer_size=500,
                                           batch_size=5,
                                           trace_length=8,
                                           tau=0.78,
                                           update_interval=nsteps,
                                           restore_model=args.restore_model,
                                           save_model=False,
                                           save_traj=True)
            elif 'GRU_DQN' == args.method:
                env = make_ple_env(args.test_env, seed=seed)
                _ = DQN.DQN_PLE.q_learning(q_network=GRU_DQN,
                                           env=env,
                                           test_env=None,
                                           seed=seed,
                                           total_timesteps=args.total_timesteps,
                                           log_interval=0,
                                           test_interval=0,
                                           show_interval=0,
                                           logdir=exp_dir,
                                           lr=args.lr,
                                           max_grad_norm=max_grad_norm,
                                           units_per_hlayer=(73, 34, 80),
                                           activ_fcn='relu6',
                                           gamma=0.85,
                                           epsilon=0.7,
                                           epsilon_decay=0.98,
                                           buffer_size=500,
                                           batch_size=5,
                                           trace_length=8,
                                           tau=0.86,
                                           update_interval=nsteps,
                                           restore_model=args.restore_model,
                                           save_model=False,
                                           save_traj=True)
            elif 'DQN' == args.method:
                env = make_ple_env(args.test_env, seed=seed)
                _ = DQN.DQN_PLE.q_learning(q_network=DQN_smac,
                                           env=env,
                                           test_env=None,
                                           seed=seed,
                                           total_timesteps=args.total_timesteps,
                                           log_interval=0,
                                           test_interval=0,
                                           show_interval=0,
                                           logdir=exp_dir,
                                           lr=args.lr,
                                           max_grad_norm=max_grad_norm,
                                           units_per_hlayer=(39, 63, 73),
                                           activ_fcn='elu',
                                           gamma=0.82,
                                           epsilon=0.35,
                                           epsilon_decay=0.990,
                                           buffer_size=500,
                                           batch_size=64,
                                           trace_length=1,
                                           tau=0.824,
                                           update_interval=nsteps,
                                           restore_model=args.restore_model,
                                           save_model=False,
                                           save_traj=True)


            env.close()


if __name__ == '__main__':
    main()
