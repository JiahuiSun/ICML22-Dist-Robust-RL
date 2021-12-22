import torch
import argparse
import time
import os

import sunblaze_envs
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import logger
from ppo import PPO
from network import FeedForwardNN
from util import set_global_seed


def main(args):
    # Configure logger
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.output, args.env_id, 'DistRobustRL', logid)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.reset()
    logger.configure(dir=log_dir)
    with open(os.path.join(log_dir, f'seed-{args.seed}.txt'), 'w') as fout:
        fout.write(f"{args}")
    set_global_seed(args.seed)

    def make_env(env_id, seed):
        def _thunk():
            env = sunblaze_envs.make(env_id)
            env.seed(seed)
            return env
        return _thunk
    env = SubprocVecEnv([make_env(env_id=args.env_id, seed=args.seed+i) for i in range(args.n_cpu)])
    env = VecNormalize(env)

    model = PPO(
        env=env,
        policy_class=FeedForwardNN,
        n_cpu=args.n_cpu,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eval_k=args.eval_k,
        block_num=args.block_num,
        repeat_per_collect=args.repeat_per_collect,
        traj_per_param=args.traj_per_param
    )

    if args.actor_model != '' and args.critic_model != '':
        print(f"Loading in {args.actor_model} and {args.critic_model}...")
        model.actor.load_state_dict(torch.load(args.actor_model))
        model.critic.load_state_dict(torch.load(args.critic_model))

    model.learn(total_iters=args.total_iters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Distribution Robust RL")
    parser.add_argument('--env_id', type=str, default='SunblazeWalker2d-v0')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--n_cpu', type=int, default=20)
    parser.add_argument('--actor_model', type=str, default='')
    parser.add_argument('--critic_model', type=str, default='')
    parser.add_argument('--output', type=str, default='output')

    parser.add_argument('--block_num', dest='block_num', type=int, default=100)
    parser.add_argument('--eval_k', dest='eval_k', type=int, default=2)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--total_iters', type=int, default=1000)
    parser.add_argument('--repeat_per_collect', type=float, default=10)
    parser.add_argument('--traj_per_param', type=float, default=1)
    args = parser.parse_args()

    main(args)
