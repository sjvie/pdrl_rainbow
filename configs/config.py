import random
import numpy as np
import torch


class Config:
    name = "DEFAULT_CONFIG"

    #######
    #       hyperparams
    #######

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.0000625

    # paper: Frames stacked: 4
    frame_stack = 4

    # paper: Action repetitions: 4
    action_repetitions = 4

    # paper: Minibatch size: 32
    batch_size = 32

    # paper: Discount factor: 0.99
    discount_factor = 0.99

    # paper: Noisy Nets sigma_o: 0.5
    noisy_sigma_zero = 0.5

    # paper: Target Network Period: 32K frames
    target_model_period = 32000

    # paper: Adam epsilon: 1.5 x 10^-4
    adam_e = 1.5e-4

    # paper: Memory size: 1M transitions
    replay_buffer_size = 1000000

    # paper: Replay period: every 4 agent steps
    replay_period = 4

    # paper: Prioritization exponent gamma: 0.5
    replay_buffer_alpha = 0.5

    # linearly increases in the rainbow paper
    # paper: Prioritization importance sampling beta: 0.4 -> 1.0
    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0
    replay_buffer_beta_annealing_steps = 1000000

    # the priority for the first experience in the replay buffer
    per_initial_max_priority = 1.0

    # paper: Distributional atoms: 51
    distributional_atoms = 51

    # paper: Distributional min/max values: [-10, 10]
    distributional_v_min = -10
    distributional_v_max = 10

    # paper: Observation down-sampling: (84, 84)
    observation_width = 84
    observation_height = 84

    # whether to use prioritized experience replay
    use_per = True

    # paper: Multi-step returns n: 3, if multistep not used, n = 1
    multi_step_n = 3

    # whether to use noisy nets
    use_noisy = True

    # whether to use distributed rl
    use_distributed = True

    # if noisy is false, you must consider epsilon greedy as exploration strategy (for now)
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000

    clip_reward = True

    #######
    #       training config
    #######

    env_name = "ALE/Breakout-v5"

    # maximum amount of episodes to train for (inf if None)
    num_episodes = None

    # maximum amount of frames to train for (inf if None)
    num_frames = None

    # maximum time to train (seconds) (inf if None)
    max_time = None

    # paper: Min history to start learning: 80K frames
    start_learning_after = 80000

    # paper: Max frames per episode: 108K
    max_frames_per_episode = 108000

    #######
    #       logging config
    #######

    log_per_frames = 100000
    log_episode_end = True
    save_video = False
    save_video_per_episodes = 10
    save_video_folder = "vid"
    save_agent_per_episodes = 1000
    agent_save_path = "agent/"
    log_file = "220613_01.log"

    loss_avg = 100
    model_log_freq = 2000

    # formatting of the log messages
    log_format = "[%(levelname)s %(asctime)s]: %(message)s"
    log_datefmt = "%y-%m-%d %H:%M:%S"

    #######
    #       miscellaneous
    #######

    # GPU Device
    gpu_device_name = "cuda:0"
    cpu_device_name = "cpu"
