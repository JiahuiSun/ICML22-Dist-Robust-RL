import json
import numpy as np
import random
import torch
from scipy.stats import multivariate_normal, uniform


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NumpyEncoder(json.JSONEncoder):
    """Ensures json.dumps doesn't crash on numpy types
    See: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class EnvParamDist():
    """Environment parameter p is a k-dimensional random variable within a given range.

    """
    def __init__(self, param_start=[0], param_end=[10], dist_type='gaussian'):
        self.start = np.array(param_start)
        self.end = np.array(param_end)
        self.mu = (self.start + self.end) / 2
        self.sigma = (self.mu - self.start) / 3
        cov = np.diag(self.sigma)**2
        if dist_type == 'gaussian':
            self.param_dist = multivariate_normal(mean=self.mu, cov=cov)
        elif dist_type == 'uniform':
            self.param_dist = uniform(loc=self.start, scale=self.end-self.start)
        else:
            raise NotImplementedError

    def set_division(self, block_num):
        traj_params, param_percents = [], {}

        block_size = (self.end - self.start) / np.sqrt(block_num)
        block_size_density = block_size.copy()
        block_size_friction = block_size.copy()
        block_size_friction[0] = 0
        block_size_density[1] = 0
        left = np.array(self.start)
        right = left + block_size_friction
        right = right + block_size_density
        param = self.sample(left, right)
        percent = self.integral(left, right)
        for N in range(block_num):
            traj_params.append(list(param[0]))
            param_percents[tuple(param[0])] = percent
            if N % block_num ** 0.5 != block_num ** 0.5 - 1:
                left = left + block_size_friction
                right = right + block_size_friction
            else:
                left = left + block_size_density
                left[1] = self.start[1]
                right = left + block_size_friction
                right = right + block_size_density
            param = self.sample(left, right)
            percent = self.integral(left, right)
        return traj_params, param_percents

    def sample(self, size=(1, 2)):
        # size = num x k
        tmp = self.param_dist.rvs(size=size)
        min_param = self.start.reshape(1, -1).repeat(size[0], axis=0)
        max_param = self.end.reshape(1, -1).repeat(size[0], axis=0)
        return np.clip(tmp, min_param, max_param)

    def integral(self, left, right):
        left_right = left.copy()
        right_left = left.copy()
        left_right[0] = right[0]
        right_left[1] = right[1]
        right_cdf = self.param_dist.cdf(right)
        left_cdf = self.param_dist.cdf(left)
        left_right_cdf = self.param_dist.cdf(left_right)
        right_left_cdf = self.param_dist.cdf(right_left)
        return right_cdf - left_right_cdf - right_left_cdf + left_cdf


class CircularList():
    def __init__(self, params=[]):
        assert params, "empty list."
        self.params = params
        self.idx = 0
        self.param2idx = {tuple(param): idx for idx, param in enumerate(params)}
        self.param_flag = np.zeros(len(self.params))
    
    def pop(self):
        param = self.params[self.idx].copy()
        self.idx = (self.idx + 1) % len(self.params)
        return param
    
    def record(self, params=[]):
        assert params, "empty list."
        finish_param_idx = [self.param2idx[tuple(param)] for param in params]
        self.param_flag[finish_param_idx] += 1

    def is_finish(self, threshold=1):
        return all(self.param_flag >= threshold)
