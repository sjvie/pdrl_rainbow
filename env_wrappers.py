import numpy as np
from gym import Wrapper, spaces


class CartPoleImageObservationWrapper(Wrapper):

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


class CartPoleIntObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        dummy_obs = self.reset()
        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.process_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.process_obs(obs)
        return obs, reward, done, info

    def process_obs(self, obs):
        obs[0] = ((obs[0] + 4.8) / 9.6) * 255
        obs[1] = sigmoid(obs[1]) * 255
        obs[2] = ((obs[2] + 0.418) / 0.836) * 255
        obs[3] = sigmoid(obs[3]) * 255
        obs = obs.clip(min=0, max=255)
        obs = obs.astype(np.uint8)
        return obs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
