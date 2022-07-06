import configs.config as config


class Config(config.Config):
    name = "DISTRIBUTED_NOISY_NORMAL_ER"

    env_name = "ALE/Pong-v5"

    use_per = False
    multi_step_n = 1
    use_noisy = True
    use_distributional = True

    clip_reward = True
