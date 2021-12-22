import numpy as np


class EnvParamSampler():
    """Environment parameter p is a k-dimensional random variable within a given range.
    
    """
    def __init__(self, param_start=[0], param_end=[10]):
        self.start = np.array(param_start)
        self.end = np.array(param_end)
        self.mu = (self.start + self.end) / 2
        self.sigma = (self.mu - self.start) / 3
        self.cov = np.diag(self.sigma)**2
    
    def clip(self, params):
        # params shape should be Nxk
        min_param = self.start.reshape(1, -1).repeat(params.shape[0], axis=0)
        max_param = self.end.reshape(1, -1).repeat(params.shape[0], axis=0)
        return np.clip(params, min_param, max_param)

    def uniform_sample(self, size=(1, 2)):
        params = np.random.uniform(low=self.start, high=self.end, size=size)
        return self.clip(params)

    def gaussian_sample(self, size=(1,)):
        params = np.random.multivariate_normal(self.mu, self.cov, size=size)
        return self.clip(params)