class Config:
    name = "RAINBOW"

    adam_learning_rate = 0.0000625
    noisy_sigma_zero = 0.5
    adam_e = 1.5e-4

    num_episodes = 100
    replay_buffer_size = 1000000

    # todo: is this the gamma exponent used in the rainbow paper?
    replay_buffer_alpha = 0.5

    # linearly increases in the rainbow paper
    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0

