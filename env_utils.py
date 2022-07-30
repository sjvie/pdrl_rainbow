import os
import time

import gym
import numpy as np

from env_wrappers import RecorderWrapper


def create_cart_pole(conf):
    env = gym.make("CartPole-v1")

    if conf.save_video_per_frames is not None:
        env = RecorderWrapper(env, conf.tmp_vid_folder, conf.save_video_per_frames, fps=30)

    conf.obs_dtype = np.float32
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n
    return env, observation_shape, action_space


def create_atari(conf):
    env = get_atari_env(conf, recorder=True)

    conf.obs_dtype = np.uint8
    observation_shape = (conf.frame_stack, conf.observation_width, conf.observation_width)
    action_space = env.action_space.n

    return env, observation_shape, action_space


def create_atari_multi(conf):
    num_envs = conf.num_envs
    env_fns = [lambda: get_atari_env(conf, recorder=True)]
    for _ in range(num_envs - 1):
        env_fns.append(lambda: get_atari_env(conf, recorder=False))

    env = gym.vector.AsyncVectorEnv(env_fns)

    conf.obs_dtype = np.uint8
    observation_shape = (conf.frame_stack, conf.observation_width, conf.observation_width)
    action_space = env.action_space[0].n
    return env, observation_shape, action_space


def get_atari_env(conf, recorder=True):
    env = gym.make(conf.env_name,
                   obs_type="grayscale",
                   full_action_space=False,
                   repeat_action_probability=conf.repeat_action_probability,
                   frameskip=1
                   )

    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=conf.max_noops,
                                          frame_skip=conf.action_repetitions,
                                          screen_size=conf.observation_width,
                                          terminal_on_life_loss=conf.terminal_on_life_loss,
                                          grayscale_obs=True
                                          )

    if conf.save_video_per_frames is not None and recorder:
        env = RecorderWrapper(env, conf.tmp_vid_folder, conf.save_video_per_frames, fps=30)

    env = gym.wrappers.FrameStack(env, conf.frame_stack)
    return env


def randomize_env(env, steps=1000):
    assert steps > 0
    for _ in range(steps - 1):
        actions = env.action_space.sample()
        env.step(actions)
    actions = env.action_space.sample()
    return env.step(actions)[0]


def get_tmp_vid_folder():
    return os.path.join("vid", "tmp_" + str(int(time.time())))
