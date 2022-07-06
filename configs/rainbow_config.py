import configs.config as config


class Config(config.Config):
    name = "RAINBOW"

    use_per = True
    multi_step_n = 3
    use_noisy = True
    use_distributional = True