import configs.config as config


class Config(config.Config):
    name = "TEST"

    replay_buffer_size = 100000
    start_learning_after = 8000

    # LOGGING
    log_per_frames = 100000
    log_episode_end = True
    save_video = False
    save_video_per_episodes = 10
    save_video_folder = "vid"
    save_agent_per_episodes = 1000
    agent_save_path = "agent/"
    log_file = "220613_01.log"

    use_per = True
    multi_step_n = 1
    use_noisy = False
    use_distributed = False

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_annealing_steps = 1000000

    clip_reward = True
