import configs.config as config


class Config(config.Config):
    name = "NO_NOISY"

    use_per = True
    multi_step_n = 3
    use_noisy = False
    use_distributional = True

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000

    clip_reward = True