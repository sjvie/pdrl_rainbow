import os
import time
from pathlib import Path

import cv2
import imageio
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


class RecorderWrapper(Wrapper):

    def __init__(self, env, save_dir, save_vid_per_episode, fps):
        super().__init__(env)
        self.save_vid_per_episode = save_vid_per_episode
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.episode = 1
        self.is_recording = False

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.episode % self.save_vid_per_episode == 0 and not self.is_recording:
            self.writer = imageio.get_writer(os.path.join(self.save_dir, str(self.episode) + ".mp4"), fps=self.fps, macro_block_size=1)
            self.is_recording = True

        if self.is_recording and not done:
            img = self.env.render("rgb_array")
            self.writer.append_data(img)

        if done:
            if self.is_recording:
                self.writer.close()
                self.is_recording = False

            self.episode += 1

        return obs, reward, done, info
