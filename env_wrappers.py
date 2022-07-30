import os
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from gym import Wrapper


class RecorderWrapper(Wrapper):

    def __init__(self, env, save_dir, save_vid_per_frames, fps):
        super().__init__(env)
        self.save_vid_per_frames = save_vid_per_frames
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.frames = 0
        self.episode = 1
        self.is_recording = False
        self.video_path = None

        self.next_video = save_vid_per_frames

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.is_recording and not done:
            img = self.env.render("rgb_array")
            self.writer.append_data(img)

        if done:
            if self.frames >= self.next_video and not self.is_recording:
                self.video_path = os.path.join(self.save_dir, str(self.episode) + ".mp4")
                self.writer = imageio.get_writer(self.video_path, fps=self.fps,
                                                 macro_block_size=1)
                self.is_recording = True
            elif self.is_recording:
                self.writer.close()
                self.is_recording = False
                self.next_video += self.save_vid_per_frames
                info["video_path"] = self.video_path

            self.episode += 1

        self.frames += 1
        return obs, reward, done, info
