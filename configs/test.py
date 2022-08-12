import configs.config as config


class Config(config.Config):
    name = "TEST"

    replay_buffer_size = 100_000
    start_learning_after = 1000
    target_model_period = 1600

    adam_learning_rate = 0.00025
    adam_e = 1.5e-4

    num_frames = 100_000

    env_name = "ALE/Breakout-v5"

    #model_arch = "impala"
    #model_arch = "d2rl"
    model_arch = "rainbow"

    model_pre_scale_factor = 2
    model_body_scale_factor = 1

    impala_adaptive_pool_size = 6

    max_noops = 0
    repeat_action_probability = 0.25
    terminal_on_life_loss = False

    batch_size = 32
    num_envs = 4
    sample_repetitions = 8

    save_video_per_frames = 2500
    save_agent_per_frames = 10_000

    per_initial_max_priority = 1

    noisy_sigma_zero = 0.5

    log_wandb = False

    cuda_deterministic = True
    seed = 1234

    distributional_v_min = -10
    distributional_v_max = 10

    distributional_atoms = 51

    use_per = True
    multi_step_n = 3
    use_noisy = True
    use_distributional = True
    use_dueling = True
    use_double = True
    use_kl_loss = True
    use_exploration = False
    use_rnd = True
    replay_buffer_alpha = 0.2

    replay_buffer_beta_start = 0.4
    replay_buffer_beta_end = 1.0
    replay_buffer_beta_annealing_steps = 60000

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 2000

    clip_reward = True

    replay_buffer_prio_offset = 1e-6
