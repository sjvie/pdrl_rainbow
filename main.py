import logging
import os
import sys
import time

import torch

import random
import train
from agent import Agent
import gym
import numpy as np

from env_wrappers import RecorderWrapper

agent_load_path = "agent/30"
config_settings = sys.argv[1]

if config_settings == 'duelling':
    from configs.duelling_config import Config
elif config_settings == 'double':
    from configs.double_config import Config
elif config_settings == 'distributional_double':
    from configs.distributional_double_config import Config
elif config_settings == 'duelling_per':
    from configs.duelling_per_config import Config
elif config_settings == 'noisy_per':
    from configs.noisy_per_config import Config
elif config_settings == 'noisy':
    from configs.noisy_config import Config
elif config_settings == 'distributional_per':
    from configs.distributional_per_config import Config
elif config_settings == 'distributional_noisy':
    from configs.distributional_noisy_config import Config
elif config_settings == 'distributional':
    from configs.distributional_config import Config
elif config_settings == 'no_noisy':
    from configs.no_noisy_config import Config
elif config_settings == 'multistep':
    from configs.multistep_config import Config
elif config_settings == 'multistep_noisy':
    from configs.multistep_noisy_config import Config
elif config_settings == 'test':
    from configs.test_config import Config
elif config_settings == 'rainbow2':
    from configs.rainbow2_config import Config
elif config_settings == 'rainbow3':
    from configs.rainbow3_config import Config
elif config_settings == 'rainbow4':
    from configs.rainbow4_config import Config
elif config_settings == 'rainbow5':
    from configs.rainbow5_config import Config
elif config_settings == 'rainbow_expl':
    from configs.rainbow_wo_per import Config
elif config_settings == 'rainbow_expl_big_batch':
    from configs.rainbow_expl_big_batch import Config
else:
    from configs.rainbow1_config import Config


def main():
    conf = Config()
    if conf.seed is None:
        conf.seed = random.randint(0, 1000)

    # set seeds for all pseudo-random number generators
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    random.seed(conf.seed)

    # CUDA uses non-deterministic methods by default
    # when toggled on in the config, CUDA is set to only use deterministic methods, reducing performance
    if conf.cuda_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    if conf.save_video_per_episodes is not None:
        conf.tmp_vid_folder = get_tmp_vid_folder()

    if conf.env_name == "cartpole":
        env, observation_shape, action_space = cart_pole(conf)
    else:
        env, observation_shape, action_space = atari_multi(conf)

    # get the correct device (either CUDA or CPU)
    conf.device = torch.device(conf.gpu_device_name if torch.cuda.is_available() else conf.cpu_device_name)

    print("Device: %s" % conf.device)
    print("seed: %s" % conf.seed)
    print("spec: %s" % env.spec)

    agent = Agent(observation_shape,
                  action_space,
                  conf
                  )

    train.train_agent(agent, env, conf=conf)


def cart_pole(conf):
    env = gym.make("CartPole-v1")

    if conf.save_video_per_episodes is not None:
        env = RecorderWrapper(env, conf.tmp_vid_folder, conf.save_video_per_episodes, fps=30)

    conf.obs_dtype = np.float32
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n
    return env, observation_shape, action_space


def atari(conf):
    env = get_atari_env(conf, recorder=True)

    conf.obs_dtype = np.uint8
    observation_shape = (conf.frame_stack, conf.observation_width, conf.observation_width)
    action_space = env.action_space.n

    return env, observation_shape, action_space


def atari_multi(conf):
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

    if conf.save_video_per_episodes is not None and recorder:
        env = RecorderWrapper(env, conf.tmp_vid_folder, conf.save_video_per_episodes, fps=30)

    env = gym.wrappers.FrameStack(env, conf.frame_stack)
    return env


def get_tmp_vid_folder():
    return os.path.join("vid", "tmp_" + str(int(time.time())))


if __name__ == "__main__":
    print("Hello, I am %s!" % Config.name)
    main()
