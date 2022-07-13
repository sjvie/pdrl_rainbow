import logging
import os
import sys

import torch

import random
import train
from agent import Agent
import gym
import numpy as np

from env_wrappers import CartPoleImageObservationWrapper, CartPoleIntObservationWrapper

agent_load_path = "agent/30"
log_file_name = "log_00.txt"
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
else:
    from configs.rainbow1_config import Config


def main():
    # todo: log stuff
    if Config.seed is None:
        Config.seed = random.randint(0, 1000)

    # set seeds for all pseudo-random number generators
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)

    # CUDA uses non-deterministic methods by default
    # when toggled on in the config, CUDA is set to only use deterministic methods, reducing performance
    if Config.cuda_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if Config.env_name == "cartpole":
        env, observation_shape, action_space = cart_pole()
    else:
        env, observation_shape, action_space = atari()

    logging.info("Cuda available: %s" % torch.cuda.is_available())
    logging.info("actionspace: %s" % action_space)
    logging.info("seed: %s" % Config.seed)
    logging.info("spec: %s" % env.spec)

    # get the correct device (either CUDA or CPU)
    device = torch.device(Config.gpu_device_name if torch.cuda.is_available() else Config.cpu_device_name)

    agent = Agent(observation_shape,
                  action_space,
                  device,
                  Config
                  )

    # agent.load(agent_load_path)
    train.train_agent(agent, env, conf=Config)


def cart_pole():
    env = gym.make("CartPole-v1")
    # env = CartPoleIntObservationWrapper(env)
    # env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    # env = gym.wrappers.FrameStack(env, Config.frame_stack)

    if Config.save_video:
        env = gym.wrappers.RecordVideo(env, Config.save_video_folder,
                                       episode_trigger=lambda x: x % Config.save_video_per_episodes == 0)

    Config.obs_dtype = np.float32
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n
    return env, observation_shape, action_space


def atari():
    env = gym.make(Config.env_name,
                   obs_type="grayscale",
                   full_action_space=False,
                   repeat_action_probability=0.0,
                   frameskip=1
                   )

    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=Config.max_noops,
                                          frame_skip=Config.action_repetitions,
                                          screen_size=Config.observation_width,
                                          terminal_on_life_loss=True,
                                          grayscale_obs=True
                                          )
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    if Config.save_video:
        env = gym.wrappers.RecordVideo(env, Config.save_video_folder,
                                       episode_trigger=lambda x: x % Config.save_video_per_episodes == 0)

    Config.obs_dtype = np.uint8
    observation_shape = (Config.frame_stack, Config.observation_width, Config.observation_width)
    action_space = env.action_space.n

    return env, observation_shape, action_space


if __name__ == "__main__":
    if Config.log_file is not None:
        logging.basicConfig(level=logging.INFO, format=Config.log_format, datefmt=Config.log_datefmt,
                            filename=Config.log_file, filemode="a")
    else:
        logging.basicConfig(level=logging.INFO, format=Config.log_format, datefmt=Config.log_datefmt)
    logging.info("Hello, I am %s!" % Config.name)
    main()
