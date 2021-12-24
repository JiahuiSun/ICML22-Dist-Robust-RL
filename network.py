import torch
from torch import nn
import numpy as np


class Critic(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_size=64):
		super(Critic, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.net = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, output_dim)
		)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float32)
		logits = self.net(obs)
		return logits


class Actor(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_size=64):
		super(Actor, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.net = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh()
		)
		self.mu = nn.Linear(64, output_dim)
		self.sigma = nn.Parameter(torch.zeros(output_dim, 1))
		torch.nn.init.constant_(self.sigma, -0.5)

	def forward(self, obs):
		"""Mapping: obs -> logits -> (mu, sigma)."""
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float32)
		logits = self.net(obs)
		mu = self.mu(logits)
		shape = [1] * len(mu.shape)
		shape[1] = -1
		sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
		return mu, sigma
