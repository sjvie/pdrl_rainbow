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
    # aka frameskip
    action_repetitions = 4

    # paper: Minibatch size: 32
    batch_size = 32

    # paper: Discount factor: 0.99
    discount_factor = 0.99

    # paper: Noisy Nets sigma_o: 0.5
    # they also proposed the possibility of lowering this to 0.1
    noisy_sigma_zero = 0.5

    # paper: Target Network Period: 32K frames
    target_model_period = 32_000

    # paper: Adam epsilon: 1.5 x 10^-4
    adam_e = 1.5e-4

    # paper: Memory size: 1M transitions
    replay_buffer_size = 1_000_000

    # average amount of times a transition is sampled
    sample_repetitions = 8

    # paper: Prioritization exponent gamma: 0.5
    replay_buffer_alpha = 0.5

    # linearly increases in the rainbow paper
    # paper: Prioritization importance sampling beta: 0.4 -> 1.0
    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0
    replay_buffer_beta_annealing_steps = 10_000_000

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

    # gradient norm clipping
    grad_clip = 10

    # whether to use prioritized experience replay
    use_per = True

    # paper: Multi-step returns n: 3, if multistep not used, n = 1
    multi_step_n = 3

    # whether to use noisy nets
    use_noisy = True

    # whether to use distributed rl
    use_distributional = True

    # whether to use dueling dqn
    use_dueling = True

    # whether to use double dqn
    use_double = True

    # whether to end the episode and reset after a life was lost
    terminal_on_life_loss = False

    # whether to use KL loss (as opposed to cross entropy loss)
    use_kl_loss = True

    # whether to use the softmax exploration
    use_exploration = False
    use_rnd = False
    
    # if noisy is false, you must consider epsilon greedy as exploration strategy (for now)
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1_000_000

    # if use_exploration is true, you will consider the softmax policy
    exp_beta_start = 0.001
    exp_beta_end = 100
    exp_beta_mid = 1
    exp_beta_annealing_steps = 1_000_000
    exp_beta_annealing_steps2 = 2_000_000

    clip_reward = True

    # the architecture of the neural network
    # rainbow: 3 conv + 2 linear
    # impala: impala conv + 2 linear
    model_arch = "rainbow"

    # scaling factor of feature/channel sizes of the pre / body layers of the neural network
    model_pre_scale_factor = 2
    model_body_scale_factor = 1

    # size of the adaptive max pool layer of the impala conv
    impala_adaptive_pool_size = 8

    # number of parallel environments
    num_envs = 64

    # amount of random actions to perform on the environments before training
    randomize_env_steps = 1000

    #######
    #       training config
    #######

    env_name = "ALE/Breakout-v5"

    # maximum amount of frames to train for (inf if -1)
    num_frames = -1

    # the maximum amount of no-ops inserted at the beginning of each episode
    # paper: up to 30
    max_noops = 0

    # probability that instead of the selected action, the environment instead uses the last action
    repeat_action_probability = 0.25

    # paper: Min history to start learning: 80K frames
    start_learning_after = 80_000

    # paper: Max frames per episode: 108K
    max_frames_per_episode = 108_000

    # whether to force cuda to act deterministic
    cuda_deterministic = False

    # The seed for pseudo-random number generation, if -1, a random seed will be selected
    seed = -1

    #######
    #       logging config
    #######

    log_wandb = True

    # these are the frames of a single agent
    # the value should be set according to num_envs
    # for num_envs = 64, a value of 1500 corresponds to approximately 100_000 frames (of all envs)
    save_video_per_frames = 1500
    save_agent_per_frames = 500_000
    save_buffer = False
    loss_avg = 500
    model_log_freq = 5000

    tmp_vid_folder = ""

    #######
    #       miscellaneous
    #######

    # GPU Device
    gpu_device_name = "cuda:0"
    cpu_device_name = "cpu"

    # replay buffer small priority offset
    replay_buffer_prio_offset = 1e-6
