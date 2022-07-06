import logging
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

# todo: multistep configs
if config_settings == 'duelling':
    from configs.duelling_config import Config
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
else:
    from configs.rainbow_config import Config


def main():
    # todo: log stuff
    seed = random.randint(0, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # config.config_benchmark()
    if Config.env_name == "cartpole":
        env, observation_shape, action_space = cart_pole()
    else:
        env, observation_shape, action_space = atari()

    # TODO: what about action repetitions?
    env.seed(seed)

    logging.info("Cuda available: %s" % torch.cuda.is_available())
    logging.info("actionspace: %s" % action_space)
    logging.info("seed: %s" % seed)
    logging.info("spec: %s" % env.spec)

    device = torch.device(Config.gpu_device_name if torch.cuda.is_available() else Config.cpu_device_name)
    agent = Agent(observation_shape,
                  action_space,
                  device,
                  seed,
                  Config
                  )

    # agent.load(agent_load_path)
    train.train_agent(agent, env, conf=Config)


def cart_pole():
    env = gym.make("CartPole-v1")
    #env = CartPoleIntObservationWrapper(env)
    # env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    # env = gym.wrappers.FrameStack(env, Config.frame_stack)

    Config.obs_dtype = torch.float32
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n
    return env, observation_shape, action_space


def atari():
    env = gym.make(Config.env_name, obs_type="grayscale", full_action_space=False, repeat_action_probability=0.0)
    env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    if Config.save_video:
        env = gym.wrappers.RecordVideo(env, Config.save_video_folder,
                                       episode_trigger=lambda x: x % Config.save_video_per_episodes == 0)

    Config.obs_dtype = torch.uint8
    observation_shape = (Config.frame_stack, Config.observation_width, Config.observation_height)
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
