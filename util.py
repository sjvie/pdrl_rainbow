import argparse
import ast
import importlib
import os
import random

import numpy as np
import torch
import wandb

import configs.config as default_config


def get_next_power_of_2(k):
    n = 1
    while n < k:
        n *= 2
    return n


def is_power_of_2(k):
    return (k & (k - 1) == 0) and k > 0


class LinearValue:
    def __init__(self, start_value, end_value, step_start, step_end):
        assert step_start < step_end

        self.start_value = start_value
        self.end_value = end_value
        self.step_start = step_start
        self.step_end = step_end

        self.total_steps = self.step_end - self.step_start
        self.diff_value = self.end_value - self.start_value

    def __call__(self, step):
        if step <= self.step_start:
            return self.start_value
        elif step >= self.step_end:
            return self.end_value
        else:
            return self.start_value + self.diff_value * ((step - self.step_start) / self.total_steps)


def init_logging(conf):
    config = {}
    for var_name in dir(conf):
        if var_name.startswith("__"):
            continue
        config[var_name] = getattr(conf, var_name)
    wandb.init(project="pdrl",
               entity="pdrl",
               mode=("online" if conf.log_wandb else "offline"),
               name=conf.name,
               config=config)


def save_agent(agent, total_frames, save_to_wandb):
    path = get_agent_save_path(total_frames)
    agent.save(path)
    if save_to_wandb:
        wandb.save(path)


def get_agent_save_path(episode):
    filename = "model_" + str(episode) + ".pt"
    return os.path.join(wandb.run.dir, filename)


def get_conf(args_raw):
    """
    Gets the config as defined by the given args.
    All possible arguments can be seen by passing the -h or --help flag.

    :param args_raw: command line args of the program
    :return: A Config object defined by the given command line args
    """
    default_conf = default_config.Config()

    default_conf_vars = {}
    for var_name in dir(default_conf):
        if var_name.startswith("__"):
            continue
        default_conf_vars[var_name] = getattr(default_conf, var_name)

    parser = argparse.ArgumentParser(description="Extended Rainbow implementation")
    parser.add_argument("-c", "--config")

    for var_name, val in default_conf_vars.items():
        parser_name = "--" + var_name
        parser.add_argument(parser_name, type=ast.literal_eval)

    args = parser.parse_args(args_raw)

    args = vars(args)
    if args["config"] is not None:
        config = importlib.import_module('configs.' + args["config"])
    else:
        config = default_config
    conf = config.Config()

    for k, v in args.items():
        if v is not None and k != "config":
            setattr(conf, k, v)

    if conf.seed == -1:
        conf.seed = random.randint(0, 1000)

    return conf


def set_determinism(seed, cuda_deterministic=False):
    """
    Sets the determinism parameters for the program.
    This sets the given seed for numpy, pytorch and random.
    Additionally, cuda can be set to use deterministic methods, reducing performance.

    :param seed: The seed to be used for pseudo-random number generation
    :param cuda_deterministic: Whether to use cuda deterministic methods
    """
    # set seeds for all pseudo-random number generators
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # CUDA uses non-deterministic methods by default
    # when toggled on in the config, CUDA is set to only use deterministic methods, reducing performance
    if cuda_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
