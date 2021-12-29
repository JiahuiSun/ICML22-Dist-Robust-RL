import torch as th
import pickle
import numpy as np
import argparse
import time
import os
from torch.distributions import Independent, Normal
from network import Actor
import sunblaze_envs
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import logger
from util import EnvParamDist
from util import set_global_seed
from ppo import PPO


class Tester(PPO):
	def __init__(
		self,
		env,
		actor,
		norm_param,
		test_params,
		dist_fn,
		param_dist='gaussian',
		add_param=True,
		device='cpu',
		action_scaling=True,
		n_cpu=4,
		traj_per_param=1
	) -> None:
		super().__init__(
			env=env, 
			actor=actor, 
			critic=None, 
			actor_optim=None, 
			critic_optim=None, 
			dist_fn=dist_fn, 
			param_dist=param_dist, 
			add_param=add_param, 
			device=device, 
			action_scaling=action_scaling, 
			n_cpu=n_cpu, 
			traj_per_param=traj_per_param
		)
		self.test_params = test_params
		self.mean = norm_param['mean']
		self.var = norm_param['var']
		self.clipob = norm_param['clipob']

	def get_action(self, obs, params):
		obs = np.clip((obs - self.mean) / np.sqrt(self.var), -self.clipob, self.clipob)
		return super().get_action(obs, params)

	def test(self):
		buffer = self.rollout(self.test_params)
		return_list = [data['return'] for param, data in buffer.items()]
		length_list = [data['length'] for param, data in buffer.items()]
		return_list.sort()
		# 指标计算
		logger.logkv('avg_return', np.mean(return_list))
		logger.logkv('wst10_return', np.mean(return_list[:len(return_list)//10]))
		logger.logkv('avg_length', np.mean(length_list))
		logger.dumpkvs()
		self.env.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Distribution Robust RL")
	parser.add_argument('--env_id', type=str, default='SunblazeWalker2d-v0')
	parser.add_argument('--seed', type=int, default=12)
	parser.add_argument('--test_params_path', type=str, default='')
	parser.add_argument('--actor_path', type=str, default='')
	parser.add_argument('--norm_param_path', type=str, default='')
	parser.add_argument('--add_param', type=int, default=1, help='whether add param')
	parser.add_argument('--action_scaling', type=int, default=1, help='whether to scale action')

	parser.add_argument('--cuda', type=int, default=-1)
	parser.add_argument('--traj_per_param', type=int, default=1)
	parser.add_argument('--param_dist', type=str, default='uniform')
	parser.add_argument('--n_cpu', type=int, default=20)
	parser.add_argument('--test_output', type=str, default='test')	
	args = parser.parse_args()

	# Configure logger and seed
	logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	log_dir = os.path.join(args.test_output, args.env_id, 'Ours', logid)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	logger.reset()
	logger.configure(dir=log_dir)
	with open(os.path.join(log_dir, f'seed-{args.seed}.txt'), 'w') as fout:
		fout.write(f"{args}")
	set_global_seed(args.seed)
	device = f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu'

	def make_env(env_id, seed):
		def _thunk():
			env = sunblaze_envs.make(env_id)
			env.seed(seed)
			return env
		return _thunk
	env = SubprocVecEnv([make_env(env_id=args.env_id, seed=args.seed+i) for i in range(args.n_cpu)])

	# 加载actor和normalization参数
	if args.add_param:
		obs_dim = env.observation_space.shape[0] + 2
	else:
		obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	actor = Actor(obs_dim, act_dim).to(device)
	actor.load_state_dict(th.load(args.actor_path))
	def dist_fn(mu, sigma):
		return Independent(Normal(mu, sigma), 1)
	with open(args.norm_param_path, 'rb') as f:
		norm_param = pickle.load(f)

	# 提前生成好的，所有人都一样
	if args.test_params_path:
		test_params = np.load(args.test_params_path)
	else:
		lower_param1, upper_param1, lower_param2, upper_param2 = env.get_lower_upper_bound()
		env_para_dist = EnvParamDist(param_start=[lower_param1, lower_param2], param_end=[upper_param1, upper_param2])
		test_params = env_para_dist.sample(size=(100,))

	tester = Tester(
		env=env,
		actor=actor,
		norm_param=norm_param,
		test_params=test_params,
		dist_fn=dist_fn,
		param_dist=args.param_dist,
		add_param=args.add_param,
		device=device,
		n_cpu=args.n_cpu,
		action_scaling=args.action_scaling,
		traj_per_param=args.traj_per_param
	)
	tester.test()
