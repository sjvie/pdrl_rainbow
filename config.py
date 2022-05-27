class Config:
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

    # todo
    # paper: ?
    num_episodes = 100

    # paper: Memory size: 1M transitions
    replay_buffer_size = 1000000

    # paper: Replay period: every 4 agent steps
    replay_period = 4

    # paper: Min history to start learning: 80K frames
    start_learning_after = 80000

    # paper: Max frames per episode: 108K
    max_frames_per_episode: 108000

    # todo: is this the gamma exponent used in the rainbow paper?
    # paper: Prioritization exponent gamma: 0.5
    replay_buffer_alpha = 0.5

    # linearly increases in the rainbow paper
    # paper: Prioritization importance sampling beta: 0.4 -> 1.0
    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0

    # paper: Multi-step returns n: 3
    multi_step_n = 3

    # paper: Distributional atoms: 51
    distributional_atoms = 51

    # paper: Distributional min/max values: [-10, 10]
    distributional_v_min = -10
    distributional_v_max = 10

    # paper: title :P
    name = "RAINBOW"

    #GPU Device
    device = "cuda:0"
