import multiprocessing as mp

import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrappers):
    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([env.step(action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'set_envparam':
                remote.send([env.unwrapped.set_envparam(*param) for env, param in zip(envs, data)])
            elif cmd == 'parameters':
                remote.send([env.unwrapped.parameters for env in envs])
            elif cmd == 'lower_upper_bound':
                remote.send(envs[0].unwrapped.lower_upper_bound)
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # NOTE: infos啥用没有，用它来保存一份reward
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), np.stack(rews)

    def reset(self, env_idx=None):
        env_idx = range(self.num_envs) if env_idx is None else env_idx
        self._assert_not_closed()
        for idx in env_idx:
            self.remotes[idx].send(('reset', None))
        obs = [self.remotes[idx].recv() for idx in env_idx]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def set_params(self, env_idx_param={}):
        self._assert_not_closed()
        for idx, param in env_idx_param.items():
            self.remotes[idx].send(('set_envparam', [param]))
        success = [self.remotes[idx].recv() for idx in env_idx_param.keys()]
        success = [tmp[0] for tmp in success]
        assert all(success), "some env fail to set params."

    def get_params(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('parameters', None))
        params = [remote.recv() for remote in self.remotes]
        params = np.array([param[0] for param in params])
        return params

    def get_lower_upper_bound(self):
        self._assert_not_closed()
        self.remotes[0].send(('lower_upper_bound', None))
        params = self.remotes[0].recv()
        return params

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]
