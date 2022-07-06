import configs.config as config


class Config(config.Config):
    name = "DISTRIBUTED_PER_NOISY"

    use_per = True
    multi_step_n = 1
    use_noisy = True
    use_distributional = True

    clip_reward = False
