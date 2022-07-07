import configs.config as config


class Config(config.Config):
    name = "DUELLING_PER"

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.0000625

    replay_buffer_alpha = 0.5

    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0
    replay_buffer_beta_annealing_steps = 10000000

    use_per = True
    multi_step_n = 1
    use_noisy = False
    use_distributional = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000

    clip_reward = True
