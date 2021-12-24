import time
import os
import numpy as np
from numba import njit
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Independent, Normal
from tensorboardX import SummaryWriter

from util import EnvParamDist
from network import Actor, Critic
from baselines import logger
from baselines.common import explained_variance


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    m = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        returns[i] = gae
    return returns


class PPO():
    def __init__(
        self, 
        env,
        n_cpu=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        vf_coef=0.25,
        block_num=100,
        repeat_per_collect=10,
        save_freq=10,
        log_freq=1
    ):
        self.env = env
        self.n_cpu = n_cpu
        self.block_num = block_num
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.vf_coef = vf_coef
        self.repeat_per_collect = repeat_per_collect
        self.save_freq = save_freq
        self.log_freq = log_freq

        self.action_space = env.action_space
        self.action_scaling = True
        self.norm_adv = True
        self.max_grad_norm = 0.5

        lower_param1, upper_param1, lower_param2, upper_param2 = self.env.get_lower_upper_bound()
        self.env_para_dist = EnvParamDist(param_start=[lower_param1, lower_param2], param_end=[upper_param1, upper_param2])

        # 创建网络并参数初始化
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim, 1)
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, th.nn.Linear):
                # orthogonal initialization
                th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                th.nn.init.zeros_(m.bias)
        for m in self.actor.mu.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
        self.optim = Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=self.lr)
        
        def dist(mu, sigma):
            return Independent(Normal(mu, sigma), 1)
        self.dist_fn = dist

    def learn(self, total_iters=1000):
        st_time = time.time()
        writer = SummaryWriter(logger.get_dir())
        traj_params = [1000, 0.8]
        for T in range(total_iters):
            # 并行sample 100个参数对应的trajectory
            buffer_obs, buffer_act, buffer_rew, buffer_done, buffer_log_prob, buffer_obs_next, buffer_real_rew = self.rollout(param=traj_params)

            buffer_return = [np.sum(buffer_real_rew[i]) for i in range(self.block_num)]
            buffer_length = [len(buffer_real_rew[i]) for i in range(self.block_num)]
 
            # 训练，每次取一条trajectory，计算GAE、loss做更新；总共100条；重复10遍
            actor_loss_list, critic_loss_list, ev_list = [], [], []
            for _ in range(self.repeat_per_collect):
                for i in range(self.block_num):
                    obs = buffer_obs[i]
                    act = buffer_act[i]
                    rew = buffer_rew[i]
                    done = buffer_done[i]
                    old_log_probs = buffer_log_prob[i]
                    obs_next = buffer_obs_next[i]
                    # Calculate loss for critic
                    vs, curr_log_probs = self.evaluate(obs, act)
                    vs_next, _ = self.evaluate(obs_next)
                    vs_numpy, vs_next_numpy = vs.detach().numpy(), vs_next.detach().numpy()
                    adv_numpy = _gae_return(
                        vs_numpy, vs_next_numpy, rew, done, self.gamma, self.gae_lambda
                    )
                    returns_numpy = adv_numpy + vs_numpy
                    returns = th.tensor(returns_numpy, dtype=th.float32)
                    critic_loss = F.mse_loss(vs, returns)

                    # Calculate loss for actor
                    adv = th.tensor(adv_numpy, dtype=th.float32)
                    old_log_probs = th.tensor(old_log_probs, dtype=th.float32)
                    if self.norm_adv:
                        adv = (adv - adv.mean()) / adv.std()
                    ratios = th.exp(curr_log_probs - old_log_probs)
                    surr1 = ratios * adv
                    surr2 = th.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
                    actor_loss = -th.min(surr1, surr2).mean()
                    loss = actor_loss + self.vf_coef * critic_loss

                    ev = explained_variance(vs_numpy, returns_numpy)
                    ev_list.append(ev)

                    self.optim.zero_grad()
                    loss.backward()
                    if self.max_grad_norm:  # clip large gradient
                        th.nn.utils.clip_grad_norm_(
                            list(self.actor.parameters())+list(self.critic.parameters()), 
                            max_norm=self.max_grad_norm
                        )
                    self.optim.step()

                    actor_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())

            # tensorboard
            writer.add_scalar('avg_return', np.mean(buffer_return), T)
            writer.add_scalar('avg_length', np.mean(buffer_length), T)
            writer.add_scalar('ev', np.mean(ev_list), T)
            writer.add_scalar('actor_loss', np.mean(actor_loss_list), T)
            writer.add_scalar('critic_loss', np.mean(critic_loss_list), T)
            if (T+1) % self.log_freq == 0:
                logger.logkv('time_elapsed', time.time()-st_time)
                logger.logkv('epoch', T)
                logger.logkv('avg_return', np.mean(buffer_return))
                logger.logkv('avg_length', np.mean(buffer_length))
                logger.logkv('ev', np.mean(ev_list))
                logger.logkv('actor_loss', np.mean(actor_loss_list))
                logger.logkv('critic_loss', np.mean(critic_loss_list))
                logger.dumpkvs()
            if (T+1) % self.save_freq == 0:
                actor_path = os.path.join(logger.get_dir(), f'actor-{T}.pth')
                critic_path = os.path.join(logger.get_dir(), f'critic-{T}.pth')
                th.save(self.actor.state_dict(), actor_path)
                th.save(self.critic.state_dict(), critic_path)
        self.env.close()

    def rollout(self, param):
        # 就给我一个参数，我给你返回N条trajectory
        buffer_obs = [[] for _ in range(self.block_num)]
        buffer_act = [[] for _ in range(self.block_num)]
        buffer_log_prob = [[] for _ in range(self.block_num)]
        buffer_rew = [[] for _ in range(self.block_num)]
        buffer_done = [[] for _ in range(self.block_num)]
        buffer_obs_next = [[] for _ in range(self.block_num)]
        buffer_real_rew = [[] for _ in range(self.block_num)]

        # 多进程并行采样，直到所有参数都被采样过
        env_idx_param = {idx: param for idx in range(self.n_cpu)}
        self.env.set_params(env_idx_param)  # set一次即可
    
        path_cnt = 0
        mb_obs, mb_rewards, mb_actions, mb_log_prob, mb_dones, mb_obs_next, mb_real_rew = [], [], [], [], [], [], []
        obs = self.env.reset()
        while True:
            actions, log_probs = self.get_action(obs)
            mb_obs.append(obs)
            mb_actions.append(actions)
            mb_log_prob.append(log_probs)

            action_remap = self.map_action(actions)
            obs_next, rewards, dones, infos = self.env.step(action_remap)
            
            mb_rewards.append(rewards)  # 这里的reward已经被归一化了
            mb_dones.append(dones)
            # 这是因为obs_next下面可能会修改
            mb_obs_next.append(obs_next.copy())
            mb_real_rew.append(infos)

            if any(dones):
                env_done_idx = np.where(dones)[0]
                obs_reset = self.env.reset(env_done_idx)
                obs_next[env_done_idx] = obs_reset
                path_cnt += sum(dones)
                if path_cnt >= self.block_num:
                    break
            obs = obs_next

        # 数据预处理
        # 把trajectory从4条轨道中分离到100条轨道
        mb_obs = np.array(mb_obs).transpose(1, 0, 2)
        mb_actions = np.array(mb_actions).transpose(1, 0, 2)
        mb_rewards = np.array(mb_rewards).transpose(1, 0)
        mb_log_prob = np.array(mb_log_prob).transpose(1, 0)
        mb_dones = np.array(mb_dones).transpose(1, 0)
        mb_obs_next = np.array(mb_obs_next).transpose(1, 0, 2)
        mb_real_rew = np.array(mb_real_rew).transpose(1, 0)

        N = 0
        exit_flag = False
        for i in range(self.n_cpu):
            pre_inds = np.where(mb_dones[i])[0]
            inds = [0]
            inds.extend(pre_inds+1)
            for st, end in zip(inds[0:-1], inds[1:]):
                buffer_obs[N] = mb_obs[i, st:end]
                buffer_act[N] = mb_actions[i, st:end]
                buffer_rew[N] = mb_rewards[i, st:end]
                buffer_done[N] = mb_dones[i, st:end]
                buffer_log_prob[N] = mb_log_prob[i, st:end]
                buffer_obs_next[N] = mb_obs_next[i, st:end]
                buffer_real_rew[N] = mb_real_rew[i, st:end]
                N += 1
                if N >= self.block_num:
                    exit_flag = True
                    break
            if exit_flag:
                break

        return buffer_obs, buffer_act, buffer_rew, buffer_done, buffer_log_prob, buffer_obs_next, buffer_real_rew

    def get_action(self, obs):
        with th.no_grad():
            mu, sigma = self.actor(obs)
            dist = self.dist_fn(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.numpy(), log_prob.numpy()

    def map_action(self, act):
        act = np.clip(act, -1.0, 1.0)
        if self.action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                "action scaling only accepts raw action range = [-1, 1]"
            low, high = self.action_space.low, self.action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def evaluate(self, batch_obs, batch_acts=None):
        vs = self.critic(batch_obs).squeeze()
        log_probs = None
        if batch_acts is not None:
            if isinstance(batch_acts, np.ndarray):
                batch_acts = th.tensor(batch_acts, dtype=th.float32)
            mu, sigma = self.actor(batch_obs)
            dist = self.dist_fn(mu, sigma)
            log_probs = dist.log_prob(batch_acts)
        return vs, log_probs
