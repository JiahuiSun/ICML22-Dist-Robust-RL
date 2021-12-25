import torch as th
import numpy as np
import argparse
import time
import os
from torch.distributions import Independent, Normal
from network import Actor, Critic
from torch.optim import Adam
import sunblaze_envs
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import logger
from ppo import PPO
from util import set_global_seed


def main(args):
    # Configure logger
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.output, args.env_id, 'PPO', logid)
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

    # 创建网络并参数初始化
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim, 1)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, th.nn.Linear):
            # orthogonal initialization
            th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            th.nn.init.zeros_(m.bias)
    for m in actor.mu.modules():
        if isinstance(m, th.nn.Linear):
            th.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    optim = Adam(list(actor.parameters())+list(critic.parameters()), lr=args.lr)
    
    def dist(mu, sigma):
        return Independent(Normal(mu, sigma), 1)

    model = PPO(
        env=env,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        n_cpu=args.n_cpu,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        block_num=args.block_num,
        repeat_per_collect=args.repeat_per_collect,
        action_scaling=args.action_scaling,
        norm_adv=args.norm_adv,
        max_grad_norm=args.max_grad_norm,
        clip=args.clip,
        save_freq=args.save_freq,
        log_freq=args.log_freq
    )

    model.learn(total_iters=args.total_iters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Distribution Robust RL")
    parser.add_argument('--env_id', type=str, default='SunblazeWalker2d-v0')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--n_cpu', type=int, default=20)
    parser.add_argument('--output', type=str, default='output')

    parser.add_argument('--block_num', type=int, default=100)
    parser.add_argument('--vf_coef', type=int, default=0.25)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--total_iters', type=int, default=100)
    parser.add_argument('--action_scaling', type=bool, default=True)
    parser.add_argument('--norm_adv', type=bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--repeat_per_collect', type=float, default=10)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    args = parser.parse_args()

    main(args)
