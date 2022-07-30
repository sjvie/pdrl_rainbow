import os
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from gym import Wrapper, spaces


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
            self.writer = imageio.get_writer(os.path.join(self.save_dir, str(self.episode) + ".mp4"), fps=self.fps,
                                             macro_block_size=1)
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
