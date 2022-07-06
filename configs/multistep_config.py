import configs.config as config


class Config(config.Config):
    name = "MULTISTEP"

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.00025 #0.0000625

    use_per = False
    multi_step_n = 3
    use_noisy = False
    use_distributional = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000
