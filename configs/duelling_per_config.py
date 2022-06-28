import configs.config as config


class Config(config.Config):
    name = "DUELLING_PER"

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.00025 #0.0000625

    use_per = True
    multi_step_n = 1
    use_noisy = False
    use_distributed = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000

    clip_reward = False
