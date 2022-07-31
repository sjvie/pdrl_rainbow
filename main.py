import sys
import torch

import env_utils
import train
import util
from agent import Agent


def main(args):
    conf = util.get_conf(args[1:])

    print("Hello, I am %s!" % conf.name)

    util.set_determinism(conf.seed, conf.cuda_deterministic)

    if conf.save_video_per_frames is not None:
        conf.tmp_vid_folder = env_utils.get_tmp_vid_folder()

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
    main(sys.argv)
