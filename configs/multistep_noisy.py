import configs.config as config


class Config(config.Config):
    name = "MULTISTEP_NOISY"

    # paper: Adam learning rate: 0.0000625
    adam_learning_rate = 0.00025 #0.0000625

    use_per = False
    multi_step_n = 3
    use_noisy = True
    use_distributed = False

    clip_reward = False
