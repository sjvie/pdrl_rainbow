import configs.config as config


class Config(config.Config):
    name = "RAINBOW3"

    use_per = True
    multi_step_n = 3
    use_noisy = True
    use_distributional = True
    use_dueling = True
    use_double = True

    noisy_sigma_zero = 0.1
    target_model_period = 32000

    env_name = "ALE/Pong-v5"