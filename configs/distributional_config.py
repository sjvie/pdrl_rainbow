import configs.config as config


class Config(config.Config):
    name = "DISTRIBUTIONAL"

    use_per = False
    multi_step_n = 1
    use_noisy = False
    use_distributional = True

    clip_reward = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000
