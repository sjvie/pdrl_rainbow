import numpy as np
from gym import Wrapper, spaces


class CartPoleObservationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        dummy_obs = self.reset()
        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = self.env.render("rgb_array")
        obs = obs.mean(-1).astype(np.uint8)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render("rgb_array")
        obs = obs.mean(-1).astype(np.uint8)
        return obs, reward, done, info
