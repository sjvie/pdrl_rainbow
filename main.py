import logging
import sys

import torch

#from test_config import Config
import random
import train
from agent import Agent
import gym
import cupy as np
agent_load_path = "agent/30"
log_file_name = "log_00.txt"
config_settings = sys.argv[1]
#todo: multistep configs
if config_settings == 'duelling':
    from configs.duelling_config import Config

elif (config_settings == 'duelling_per'):
    from configs.duelling_per_config import Config
elif (config_settings == 'duelling_per'):
    from configs.noisy_per_config import Config
elif (config_settings == 'duelling_per'):
    from configs.distributed_per_config import Config
else:
    from configs.rainbow_config import Config

def main():

    # todo: log stuff
    seed = random.randint(0,100)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # config.config_benchmark()
    env, observation_shape, action_space = atari()
    # TODO: what about action repetitions?
    env.seed(seed)

    logging.info("Cuda available: %s" % torch.cuda.is_available())
    logging.info("actionspace: %s" % action_space)
    logging.info("seed: %s" % seed)
    agent = Agent(observation_shape,
                  Config.frame_stack,
                  action_space,
                  Config.distributional_atoms,
                  Config.distributional_v_min,
                  Config.distributional_v_max,
                  Config.discount_factor,
                  Config.batch_size,
                  Config.multi_step_n,
                  Config.tensor_replay_buffer,
                  Config.use_per,
                  Config.use_multistep,
                  Config.noisy,
                  Config.epsilon,
                  Config.epsilon_min,
                  Config.distributed,
                  seed,
                  Config.adam_learning_rate,
                  Config.adam_e,
                  Config.replay_buffer_beta_start,
                  Config.replay_buffer_alpha)

    # agent.load(agent_load_path)
    train.train_agent(agent, env, conf=Config)


def atari():
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale",full_action_space=False,repeat_action_probability=0.0)
    env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    if Config.save_video:
        env = gym.wrappers.RecordVideo(env, Config.save_video_folder,
                                       episode_trigger=lambda x: x % Config.save_video_per_episodes == 0)

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
