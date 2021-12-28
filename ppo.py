import pickle
import time
import os
import numpy as np
from numba import njit
import torch as th
import torch.nn.functional as F

from util import EnvParamDist, CircularList
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
        actor,
        critic,
        actor_optim,
        critic_optim,
        dist_fn,
        n_cpu=4,
        eval_k=2,
        traj_per_param=1,
        gamma=0.99,
        gae_lambda=0.95,
        param_dist='uniform',
        block_num=100,
        repeat_per_collect=10,
        action_scaling=True,
        norm_adv=True,
        add_param=True,
        max_grad_norm=0.5,
        clip=0.2,
        save_freq=10,
        log_freq=1,
        writer=None,
        device='cpu'
    ):
        self.env = env
        self.n_cpu = n_cpu
        self.eval_k = eval_k
        self.traj_per_param = traj_per_param
        self.block_num = block_num
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.param_dist = param_dist
        self.repeat_per_collect = repeat_per_collect
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.writer = writer
        self.device = device

        self.action_space = env.action_space
        self.action_scaling = action_scaling
        self.norm_adv = norm_adv
        self.add_param = add_param
        self.max_grad_norm = max_grad_norm

        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.dist_fn = dist_fn

        lower_param1, upper_param1, lower_param2, upper_param2 = self.env.get_lower_upper_bound()
        self.env_para_dist = EnvParamDist(
            param_start=[lower_param1, lower_param2], 
            param_end=[upper_param1, upper_param2], 
            dist_type=param_dist
        )

    def learn(self, total_iters=1000):
        st_time = time.time()
        writer = self.writer
        for T in range(total_iters):
            st1 = time.time()
            # 一次性返回100个参数及其概率
            traj_params, param_percents = self.env_para_dist.set_division(self.block_num)
            # 并行sample 100个参数对应的trajectory
            buffer = self.rollout(traj_params=traj_params)
            # 根据每个参数的return从小到大排序，并计算weight
            seq_dict_list = [
                {
                    'param': param, 'return': data['return'], 
                    'length': data['length'], 'percent': param_percents[param]
                } 
                for param, data in buffer.items()
            ]
            seq_dict_list = sorted(seq_dict_list, key=lambda e: e['return'])
            x = 0
            for j in range(self.block_num):
                y = x + seq_dict_list[j]['percent']
                seq_dict_list[j]['weight'] = (1-x)**self.eval_k - (1-y)**self.eval_k
                x = y
            traj_return = np.array([traj['return'] for traj in seq_dict_list])
            traj_length = np.array([traj['length'] for traj in seq_dict_list])
            traj_weight_dict = {traj['param']: traj['weight'] for traj in seq_dict_list}
            traj_weight = np.array([traj['weight'] for traj in seq_dict_list])
            st2 = time.time()

            # training
            actor_loss_list, critic_loss_list, ev_list = [], [], []
            for _ in range(self.repeat_per_collect):
                total_actor_loss, total_critic_loss = 0, 0
                for param, data in buffer.items():
                    # Calculate loss for critic
                    vs, curr_log_probs = self.evaluate(data['obs'], np.array(param), data['act'])
                    vs_next, _ = self.evaluate(data['obs_next'], np.array(param))
                    vs_numpy, vs_next_numpy = vs.detach().numpy(), vs_next.detach().numpy()
                    adv_numpy = _gae_return(
                        vs_numpy, vs_next_numpy, data['rew'], data['done'], self.gamma, self.gae_lambda
                    )
                    returns_numpy = adv_numpy + vs_numpy
                    returns = th.tensor(returns_numpy, dtype=th.float32, device=self.device)
                    critic_loss = F.mse_loss(vs, returns)

                    # Calculate loss for actor
                    adv = th.tensor(adv_numpy, dtype=th.float32, device=self.device)
                    old_log_probs = th.tensor(data['log_prob'], dtype=th.float32, device=self.device)
                    if self.norm_adv:
                        adv = (adv - adv.mean()) / adv.std()
                    ratios = th.exp(curr_log_probs - old_log_probs)
                    surr1 = ratios * adv
                    surr2 = th.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
                    actor_loss = -th.min(surr1, surr2).mean()
                    
                    total_actor_loss += traj_weight_dict[param] * actor_loss
                    total_critic_loss += 1 / self.block_num * critic_loss
                    ev = explained_variance(vs_numpy, returns_numpy)
                    ev_list.append(ev)

                self.actor_optim.zero_grad()
                total_actor_loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                total_critic_loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    th.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                self.critic_optim.step()

                actor_loss_list.append(total_actor_loss.item())
                critic_loss_list.append(total_critic_loss.item())

            end_time = time.time()
            # log everything
            writer.add_scalar('metric/avg_return', np.mean(traj_return), T)
            writer.add_scalar('metric/wst10_return', np.mean(traj_return[:len(traj_return)//10]), T)
            writer.add_scalar('metric/avg_length', np.mean(traj_length), T)
            writer.add_scalar('metric/E_bar', np.sum(traj_return*traj_weight), T)
            writer.add_scalar('loss/ev', np.mean(ev_list), T)
            writer.add_scalar('loss/actor_loss', np.mean(actor_loss_list), T)
            writer.add_scalar('loss/critic_loss', np.mean(critic_loss_list), T)
            if (T+1) % self.log_freq == 0:
                logger.logkv('time_rollout', st2-st1)
                logger.logkv('time_training', end_time-st2)
                logger.logkv('time_one_epoch', end_time-st1)
                logger.logkv('time_elapsed', time.time()-st_time)
                logger.logkv('epoch', T)
                logger.logkv('avg_return', np.mean(traj_return))
                logger.logkv('wst10_return', np.mean(traj_return[:len(traj_return)//10]))
                logger.logkv('avg_length', np.mean(traj_length))
                logger.logkv('E_bar', np.sum(traj_return*traj_weight))
                logger.logkv('ev', np.mean(ev_list))
                logger.logkv('actor_loss', np.mean(actor_loss_list))
                logger.logkv('critic_loss', np.mean(critic_loss_list))
                logger.dumpkvs()
            if (T+1) % self.save_freq == 0:
                actor_path = os.path.join(logger.get_dir(), f'actor-{T}.pth')
                critic_path = os.path.join(logger.get_dir(), f'critic-{T}.pth')
                th.save(self.actor.state_dict(), actor_path)
                th.save(self.critic.state_dict(), critic_path)
                traj_info_path = os.path.join(logger.get_dir(), f'seq_dict_list-{T}.pkl')
                with open(traj_info_path, 'wb') as fout:
                    pickle.dump(seq_dict_list, fout)
                # 保存obs的normalization参数
                obs_norms = {
                    'clipob': self.env.clipob,
                    'mean': self.env.ob_rms.mean,
                    'var': self.env.ob_rms.var+self.env.epsilon
                }
                norm_param_path = os.path.join(logger.get_dir(), f'norm_param-{T}.pkl')
                with open(norm_param_path, 'wb') as f:
                    pickle.dump(obs_norms, f)
        self.env.close()

    def rollout(self, traj_params=[]):
        # 每个参数收集若干条trajectory
        buffer = {tuple(param): {'obs': [], 
                                 'act': [], 
                                 'log_prob': [], 
                                 'rew': [], 
                                 'done': [],
                                 'obs_next': [],
                                 'real_rew': []
                                } for param in traj_params}
        traj_params = CircularList(traj_params)

        # 多进程并行采样，直到所有参数都被采样过
        env_idx_param = {idx: traj_params.pop() for idx in range(self.n_cpu)}
        self.env.set_params(env_idx_param)
        obs = self.env.reset()
        while True:
            params = np.array(self.env.get_params())
            actions, log_probs = self.get_action(obs, params)
            mapped_actions = self.map_action(actions)
            obs_next, rewards, dones, infos = self.env.step(mapped_actions)
            
            for idx, param in env_idx_param.items():
                buffer[tuple(param)]['obs'].append(obs[idx])
                buffer[tuple(param)]['act'].append(actions[idx])
                buffer[tuple(param)]['log_prob'].append(log_probs[idx])
                buffer[tuple(param)]['rew'].append(rewards[idx])  # 这里的reward已经被归一化了
                buffer[tuple(param)]['done'].append(dones[idx])
                buffer[tuple(param)]['obs_next'].append(obs_next[idx].copy())
                buffer[tuple(param)]['real_rew'].append(infos[idx])

            if any(dones):
                env_done_idx = np.where(dones)[0]
                # 采样停止条件：每个param都采样完一条trajectory，可以修改
                traj_params.record([env_idx_param[idx] for idx in env_done_idx])
                if traj_params.is_finish(threshold=self.traj_per_param):
                    break
                env_new_param = {idx: traj_params.pop() for idx in env_done_idx}
                self.env.set_params(env_new_param)
                obs_reset = self.env.reset(env_done_idx)
                obs_next[env_done_idx] = obs_reset
                env_idx_param.update(env_new_param)
            obs = obs_next

        # 数据预处理
        for param, data in buffer.items():
            data['obs'] = np.array(data['obs'])
            data['act'] = np.array(data['act'])
            data['log_prob'] = np.array(data['log_prob'])
            data['rew'] = np.array(data['rew'])
            data['done'] = np.array(data['done'])
            data['obs_next'] = np.array(data['obs_next'])
            data['real_rew'] = np.array(data['real_rew'])
            done_idx = np.where(data['done'])[0]
            data['return'] = np.sum(data['real_rew'][:max(done_idx)+1]) / len(done_idx)
            data['length'] = (max(done_idx)+1) / len(done_idx)
        return buffer

    def get_action(self, obs, params):
        if self.add_param:
            params = self.norm_params(params)
            obs_params = np.concatenate((obs, params), axis=1)
        else:
            obs_params = obs
        obs_params = th.tensor(obs_params, dtype=th.float32, device=self.device)
        with th.no_grad():
            mu, sigma = self.actor(obs_params)
            dist = self.dist_fn(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def norm_params(self, params):
        # input: Nx2, output: Nx2
        mu, sigma = self.env_para_dist.mu, self.env_para_dist.sigma
        return (params - mu) / sigma

    def map_action(self, act):
        act = np.clip(act, -1.0, 1.0)
        if self.action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                "action scaling only accepts raw action range = [-1, 1]"
            low, high = self.action_space.low, self.action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def evaluate(self, batch_obs, param, batch_acts=None):
        if self.add_param:
            batch_param = param.reshape(-1, 2).repeat(batch_obs.shape[0], axis=0)
            batch_param = self.norm_params(batch_param)
            batch_obs_params = np.concatenate((batch_obs, batch_param), axis=1)
        else:
            batch_obs_params = batch_obs
        batch_obs_params = th.tensor(batch_obs_params, dtype=th.float32, device=self.device)
        vs = self.critic(batch_obs_params).squeeze()
        log_probs = None
        if batch_acts is not None:
            if isinstance(batch_acts, np.ndarray):
                batch_acts = th.tensor(batch_acts, dtype=th.float32, device=self.device)
            mu, sigma = self.actor(batch_obs_params)
            dist = self.dist_fn(mu, sigma)
            log_probs = dist.log_prob(batch_acts)
        return vs, log_probs
