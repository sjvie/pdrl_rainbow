import configs.config as config


class Config(config.Config):
    name = "Montezuma rainbow epsilon 10M"

    use_per = True
    multi_step_n = 3
    use_noisy = False
    use_distributional = True
    use_dueling = True
    use_double = True
    use_kl_loss = True

    model_arch = "rainbow"
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

    loss_avg = 50

    num_frames = 10_000_000

    env_name = "ALE/MontezumaRevenge-v5"
