#!/usr/local/bin/python3.6

from run_ple_utils import make_ple_envs, arg_parser
from models_OAI import MlpPolicy, FCPolicy, CastaPolicy, LargerMLPPolicy
from A2C_OAI_NENVS import learn

def main():
    parser = arg_parser()
    parser.add_argument('--nenvs', help='Number of environments', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'casta', 'largemlp'], default='mlp')
    parser.add_argument('--nsteps', help='n environment steps per train update', type=int, default=50)
    parser.add_argument('--vf_coeff', help='Weight of value function loss in total loss', type=float, default=0.2)
    parser.add_argument('--ent_coeff', help='Weight of entropy in total loss', type=float, default=1e-7)
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-5)
    parser.add_argument('--lrschedule', help='Learning Rate Decay Schedule', choices=['constant', 'linear', 'double_linear_con'], default='constant')
    parser.add_argument('--max_grad_norm', help='Maximum gradient norm up to which gradient is not clipped', type=float, default=0.01)
    parser.add_argument('--log_interval', help='parameter values stored in tensorboard summary every <log_interval> model update step. 0 --> no logging ', type=int, default=30)
    parser.add_argument('--save_interval', help='Model is saved after <save_interval> model updates', type=int, default=1000)
    args = parser.parse_args()

    seed = args.seed
    print(args.env, args.nenvs)
    env = make_ple_envs(args.env, num_env=args.nenvs, seed=seed)
    print(env)

    if args.policy == 'mlp':
        policy_fn = MlpPolicy
    elif args.policy == 'casta':
        policy_fn = CastaPolicy
    elif args.policy == 'largemlp':
        policy_fn = LargerMLPPolicy
    learn(policy_fn, env, seed=seed, nsteps=args.nsteps, vf_coef=args.vf_coeff, ent_coef=args.ent_coeff, gamma=args.gamma,
          lr=args.lr, lrschedule=args.lrschedule, max_grad_norm=args.max_grad_norm, log_interval=args.log_interval,
          save_interval=args.save_interval, total_timesteps=args.total_timesteps)
    env.close()


if __name__ == '__main__':
    main()
