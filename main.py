import logging
import time

import numpy as np
import torch

import config
import train
from agent import Agent
from config import Config
import gym

agent_save_path = "agent/"
agent_load_path = "agent/30"
log_file_name = "log_00.txt"


def main():
    # todo: log stuff

    config.config_benchmark()
    env, observation_shape, action_space = pong()
    # TODO: what about action repetitions?

    logging.debug("Cuda available: %s" % torch.cuda.is_available())

    agent = Agent(observation_shape,
                  Config.frame_stack,
                  action_space,
                  Config.distributional_atoms,
                  Config.distributional_v_min,
                  Config.distributional_v_max,
                  Config.discount_factor,
                  Config.batch_size,
                  Config.multi_step_n,
                  Config.tensor_replay_buffer)

    # agent.load(agent_load_path)
    train.train_agent(agent, env, num_episodes=Config.num_episodes, num_frames=Config.num_frames, conf=Config)


def pong():
    # env = gym_wrappers.make_atari("ALE/Pong-v5")
    # env = gym_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=True)

    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    if Config.save_video:
        env = gym.wrappers.RecordVideo(env, Config.save_video_folder,
                                       episode_trigger=lambda x: x % Config.save_video_per_episodes == 0)

    observation_shape = (Config.frame_stack, Config.observation_width, Config.observation_height)
    action_space = env.action_space.n

    return env, observation_shape, action_space


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=Config.log_format, datefmt=Config.log_datefmt)
    logging.info("Hello, I am %s!" % Config.name)
    main()
