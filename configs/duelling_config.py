import random
import numpy as np
import torch


class Config:

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.00025 #0.0000625

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

    # todo
    # paper: ?
    num_episodes = 8000000

    # maximum amount of total frames (for training)
    num_frames = None

    # maximum time to train
    max_time = 30

    # paper: Memory size: 1M transitions
    replay_buffer_size = 100000

    # paper: Replay period: every 4 agent steps
    replay_period = 4

    # paper: Min history to start learning: 80K frames
    start_learning_after = 80000

    # paper: Max frames per episode: 108K
    max_frames_per_episode = 108000

    # paper: Prioritization exponent gamma: 0.5
    replay_buffer_alpha = 0.5

    # linearly increases in the rainbow paper
    # paper: Prioritization importance sampling beta: 0.4 -> 1.0
    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0

    # paper: Distributional atoms: 51
    distributional_atoms = 51

    # paper: Distributional min/max values: [-10, 10]
    distributional_v_min = -10
    distributional_v_max = 10

    # paper: Observation down-sampling: (84, 84)
    observation_width = 84
    observation_height = 84

    # paper: title :P
    name = "RAINBOW"

    # GPU Device
    device = "cuda:0"

    # Whether to store the replay buffer as torch tensors (as opposed to numpy arrays)
    tensor_replay_buffer = True

    # LOGGING
    log_per_frames = 100000
    log_episode_end = True
    save_video = False
    save_video_per_episodes = 10
    save_video_folder = "vid"
    save_agent_per_episodes = 1000
    agent_save_path = "agent/"
    log_file = "220613_01.log"

    # formatting of the log messages
    log_format = "[%(levelname)s %(asctime)s]: %(message)s"
    log_datefmt = "%y-%m-%d %H:%M:%S"

    #set per & multistep to false if Duelling
    #prioritized Replay
    use_per = False

    #multistep return, if False, don't forget to change multi_step_n to 1 !!!
    use_multistep = False

    # paper: Multi-step returns n: 3, if multistep not used, n = 1
    multi_step_n = 1
    #set noisy net experience
    noisy = False
    #use distributed rl
    distributed = False
    #if noisy is false, you must consider epsilon greedy as exploration strategy(for now)
    epsilon = 1
    epsilon_min = 0.01

def config_benchmark():
    Config.log_per_frames = 1000000
    Config.log_episode_end = True
    Config.save_video = False
    Config.start_learning_after = 100
    Config.replay_buffer_size = 100000
    Config.num_episodes = None
    Config.num_frames = 3000
    Config.max_time = None
    Config.tensor_replay_buffer = True
    Config.log_file = None
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

