import configs.config as config


class Config(config.Config):
    name = "rainbow_expl"

    use_per = False
    multi_step_n = 3
    use_noisy = True
    use_distributional = False
    use_dueling = True
    use_double = True
    use_exploration = True
    noisy_sigma_zero = 0.5
    target_model_period = 32000
    adam_learning_rate = 0.00025
    exp_beta_start = 0.001
    exp_beta_end = 100
    exp_beta_mid = 1
    exp_beta_annealing_steps = 1000000
    exp_beta_annealing_steps2 = 3000000

    env_name = "ALE/Breakout-v5"
