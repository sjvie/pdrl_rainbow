import configs.config as config


class Config(config.Config):
    name = "DUELLING_NOISY_PER"

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.00025 #0.0000625

    use_per = True
    multi_step_n = 1
    use_noisy = True
    use_distributed = False

    clip_reward = False
