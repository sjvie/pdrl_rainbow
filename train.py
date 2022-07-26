import os
import time
import numpy as np
import wandb


def train_agent(agent, env, conf):
    total_frames = 0
    train_frames = 0

    # for logging
    loss_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    weight_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    action_amounts = np.zeros((agent.action_space,), dtype=np.int32)
    action_distribution_log_names = ["action_" + str(x) for x in range(agent.action_space)]

    episode = 1
    if conf.num_episodes is not None:
        end_episode = episode + conf.num_episodes
    else:
        end_episode = None

    init_logging(conf)
    wandb.watch(agent.model, log='all', log_freq=conf.model_log_freq)

    print("Starting training")
    start_time = time.time()
    # uniform action distribution
    action_prob = np.ones(agent.action_space) * (1 / agent.action_space)
    distribution = np.empty(agent.action_space)
    beta = 0
    while (end_episode is None or episode <= end_episode) \
            and (conf.num_frames is None or total_frames < conf.num_frames) \
            and (conf.max_time is None or time.time() < start_time + conf.max_time):

        state = env.reset(seed=conf.seed)
        state = process_state(state)

        episode_over = False
        episode_frames = 0
        episode_reward = 0
        modified_reward = 0
        episode_loss = 0
        episode_start_time = time.time()

        while not episode_over and episode_frames < conf.max_frames_per_episode and (
                conf.num_frames is None or total_frames < conf.num_frames):
            total_frames += 1
            episode_frames += 1

            frame_log = {}

            if conf.use_exploration:
                action, beta, log_ratio = agent.select_action(state, action_prob)
            else:
                action = agent.select_action(state, action_prob)
            action_amounts[action] += 1

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            # we do not clip the total with this softmax exploration
            if conf.use_exploration:
                episode_reward += reward
                reward += reward - (1 / beta) * log_ratio
                modified_reward += reward
            else:
                episode_reward += reward
                if conf.clip_reward:
                    reward = np.clip(reward, -1, 1)

            agent.step(state, action, reward, done)

            if total_frames > conf.start_learning_after and total_frames % conf.replay_period == 0:
                loss, weights = agent.train()
                loss = loss.cpu().detach().numpy()
                if conf.use_per:
                    weights = weights.cpu().detach().numpy()

                if train_frames % conf.loss_avg == 0 and train_frames >= conf.loss_avg:
                    frame_log["frame_loss_avg"] = loss_list.mean()
                    frame_log["frame_loss_min"] = loss_list.min()
                    frame_log["frame_loss_max"] = loss_list.max()
                    if conf.use_per:
                        frame_log["buffer_tree_sum"] = agent.replay_buffer.tree.sum()
                        frame_log["buffer_tree_min"] = agent.replay_buffer.tree.min()
                        frame_log[
                            "buffer_max_priority_with_alpha"] = agent.replay_buffer.max_priority ** agent.replay_buffer.alpha
                        frame_log["frame_weights_avg"] = weight_list.mean()
                    if conf.use_exploration:
                        frame_log["exploration_beta"] = agent.exp_beta
                        frame_log["modified_reward"] = modified_reward

                # save for logging
                loss_list[train_frames % conf.loss_avg] = loss
                if conf.use_per:
                    weight_list[train_frames % conf.loss_avg] = weights

                episode_loss += loss.sum()
                train_frames += 1

            if total_frames > conf.start_learning_after \
                    and total_frames % conf.target_model_period == 0 \
                    and conf.use_double:
                agent.update_target_model()

            if total_frames % conf.save_agent_per_frames == 0 and total_frames > 0:
                save_agent(agent, total_frames, conf.log_wandb)

            state = next_state
            episode_over = done

            wandb.log(frame_log, step=total_frames)

        episode_end_time = time.time()
        fps = episode_frames / max(episode_end_time - episode_start_time, 0.0001)

        episode_log = {}
        if conf.save_video_per_episodes is not None and episode % conf.save_video_per_episodes == 0:
            episode_log["video"] = wandb.Video(os.path.join(conf.tmp_vid_folder, str(episode) + ".mp4"))

        episode_log["episode_fps"] = fps
        episode_log["episode_reward"] = episode_reward
        episode_log["episode_length"] = episode_frames
        action_distribution_dict = dict(zip(action_distribution_log_names, action_amounts / action_amounts.sum()))
        episode_log.update(action_distribution_dict)
        episode_log["episode_reward"] = episode_reward
        if not conf.use_noisy:
            episode_log["episode_exploration_rate"] = agent.epsilon
        if conf.use_per:
            episode_log["episode_per_beta"] = agent.replay_buffer.beta

        wandb.log(episode_log, step=total_frames)

        action_amounts.fill(0)

        episode += 1

    end_time = time.time()
    t = end_time - start_time
    hours = int((end_time - start_time) // 3600)
    t -= hours * 3600
    minutes = int(t // 60)
    t -= minutes * 60
    seconds = t
    print("Training finished")
    print("Trained for {} frames in {:02d}:{:02d}:{:02.2f}".format(total_frames, hours, minutes, seconds))


def init_logging(conf):
    wandb.init(project="pdrl", entity="pdrl", mode=("online" if conf.log_wandb else "offline"),
               config={
                   "config_name": conf.name,
                   "adam_learning_rate": conf.adam_learning_rate,
                   "adam_eps": conf.adam_e,
                   "discount_factor": conf.discount_factor,
                   "noisy_net_sigma": conf.noisy_sigma_zero,
                   "replay_buffer_size": conf.replay_buffer_size,
                   "replay_buffer_alpha": conf.replay_buffer_alpha,
                   "replay_buffer_beta": {"start": conf.replay_buffer_beta_start,
                                          "end": conf.replay_buffer_beta_end,
                                          "annealing_steps": conf.replay_buffer_beta_annealing_steps},
                   "per_initial_max_priority": conf.per_initial_max_priority,
                   "distributional_atoms": conf.distributional_atoms,
                   "epsilon": {"start": conf.epsilon_start,
                               "end": conf.epsilon_end,
                               "annealing_steps": conf.epsilon_annealing_steps},
                   "clip_reward": conf.clip_reward,
                   "seed": conf.seed,
                   "use_per": conf.use_per,
                   "use_distributed": conf.use_distributional,
                   "multi_step_n": conf.multi_step_n,
                   "use_noisy": conf.use_noisy,
                   "loss_avg": conf.loss_avg,
                   "start_learning_after": conf.start_learning_after,
                   "device": conf.device,
                   "env": conf.env_name,
                   "target_model_period": conf.target_model_period,
                   "cuda_deterministic": conf.cuda_deterministic,
                   "use_kl_loss": conf.use_kl_loss,
                   "use_dueling": conf.use_dueling,
                   "use_double": conf.use_double,
                   "grad_clip": conf.grad_clip,
               })


def save_agent(agent, total_frames, save_to_wandb):
    path = get_agent_save_path(total_frames)
    agent.save(path)
    if save_to_wandb:
        wandb.save(path)


def get_agent_save_path(episode):
    filename = "model_" + str(episode) + ".pt"
    return os.path.join(wandb.run.dir, filename)


def process_state(state):
    return np.array(state).squeeze()
