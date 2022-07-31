import configs.config as config


class Config(config.Config):
    name = "DISTRIBUTIONAL_DOUBLE"

    # as stated in the distributional paper
    adam_learning_rate = 0.00025

    # as stated in the distributional paper
    # (0.01 / batch_size)
    adam_e = 0.0003125


    use_per = False
    multi_step_n = 1
    use_noisy = False
    use_distributional = True
    use_dueling = False

    clip_reward = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000
