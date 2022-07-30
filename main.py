import logging
import os
import sys
import time

import torch

import random

import env_utils
import train
from agent import Agent
import numpy as np

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
        conf.tmp_vid_folder = env_utils.get_tmp_vid_folder()

    if conf.env_name == "cartpole":
        env, observation_shape, action_space = env_utils.create_cart_pole(conf)
    else:
        env, observation_shape, action_space = env_utils.create_atari_multi(conf)

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


if __name__ == "__main__":
    print("Hello, I am %s!" % Config.name)
    main()
