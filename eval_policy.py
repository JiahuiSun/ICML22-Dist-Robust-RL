import torch as th
import pickle
import numpy as np
import argparse
import time
import os
from torch.distributions import Independent, Normal
from network import Actor
import sunblaze_envs
from baselines import logger
from util import EnvParamDist
from util import set_global_seed


def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

def test(args):
	# TODO: 你必须test，不然都不知道结果好坏
	# set log and seed
	set_global_seed(args.seed)

	# 创建环境
	env = sunblaze_envs.make(args.env_id)
	env.seed(args.seed)

	# sample 100个参数
	lower_param1, upper_param1, lower_param2, upper_param2 = env.unwrapped.get_lower_upper_bound()
	env_para_dist = EnvParamDist(param_start=[lower_param1, lower_param2], param_end=[upper_param1, upper_param2])
	test_params = env_para_dist.sample(size=(100, 2))

	# 先把模型、norm参数加载进来
	if args.add_param:
		obs_dim = env.observation_space.shape[0] + 2
	else:
		obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	actor = Actor(obs_dim, act_dim)
	actor.load_state_dict(th.load(args.actor_path))
	with open(args.norm_param_path, 'rb') as f:
		norm_param = pickle.load(f)
	def dist_fn(mu, sigma):
		return Independent(Normal(mu, sigma), 1)

	# 啥都有了，开始仿真吧
	# sample 100个参数，然后跑一遍
	# 如果所有的人可以同时加载参数
	return_dict = {}
	length_dict = {}
	for param1, param2 in test_params:
		Jpi = 0
		env.unwrapped.set_envparam(param1, param2)
		obs = env.reset()
		for t in range(env.max_episode_steps):
			obs = np.clip((obs - norm_param['mean']) / np.sqrt(norm_param['var'] + norm_param['epsilon']), -norm_param['clipob'], norm_param['clipob'])
			with th.no_grad():
				mu, sigma = actor(obs)
				dist = dist_fn(mu, sigma)
				act = dist.sample()
			obs_next, rew, done, _ = env.step(act.numpy())
			Jpi += rew
			if done:
				break
			obs = obs_next
		return_dict[tuple(param1, param2)] = Jpi
		length_dict[tuple(param1, param2)] = t+1


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Distribution Robust RL")
	parser.add_argument('--env_id', type=str, default='SunblazeWalker2d-v0')
	parser.add_argument('--seed', type=int, default=12)
	parser.add_argument('--output', type=str, default='output')
	parser.add_argument('--block_num', type=int, default=100)
	parser.add_argument('--add_param', type=int, default=1, help='whether add param')
	parser.add_argument('--log_freq', type=int, default=1)
	parser.add_argument('--save_freq', type=int, default=10)
	args = parser.parse_args()

	test(args)
