import configs.config as config


class Config(config.Config):
    name = "Asterix Impala dist new_env 10M"

    use_per = True
    multi_step_n = 3
    use_noisy = True
    use_distributional = False
    use_dueling = True
    use_double = True
    use_kl_loss = True

    model_arch = "impala"
    model_pre_scale_factor = 2
    model_body_scale_factor = 1
    impala_adaptive_pool_size = 8

    batch_size = 256
    num_envs = 64
    sample_repetitions = 8

    max_noops = 0
    repeat_action_probability = 0.25
    terminal_on_life_loss = False

    adam_learning_rate = 0.00025
    adam_e = 0.0000195
    # adam_learning_rate = 0.0000625
    # adam_e = 1.5e-4

    use_exploration = True
    exp_beta_start = 0.001
    exp_beta_end = 100
    exp_beta_mid = 1
    exp_beta_annealing_steps = 1000000
    exp_beta_annealing_steps2 = 3000000

    loss_avg = 50

    num_frames = 10_000_000

    env_name = "ALE/Asterix-v5"
