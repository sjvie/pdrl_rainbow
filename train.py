import os
import time
import numpy as np
import torch

import wandb


def train_agent(agent, env, conf):
    num_envs = conf.num_envs
    # these values need to be a multiple of the number of parallel envs to work correctly
    assert conf.target_model_period % num_envs == 0
    assert conf.save_agent_per_frames % num_envs == 0
    assert conf.sample_repetitions % num_envs == 0 or num_envs % conf.sample_repetitions == 0
    assert conf.batch_size % num_envs == 0 or num_envs % conf.batch_size == 0
    assert conf.batch_size % conf.sample_repetitions == 0 or conf.sample_repetitions % conf.batch_size == 0

    if num_envs * conf.sample_repetitions > conf.batch_size:
        train_reps = (num_envs * conf.sample_repetitions) // conf.batch_size
        train_per = 1
    else:
        train_reps = 1
        train_per = (conf.batch_size // conf.sample_repetitions) // num_envs

    conf.train_reps = train_reps
    conf.train_per = train_per

    total_frames = 0
    train_frames = 0
    episode = 1
    video_episode = 1

    # for logging
    loss_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    weight_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)

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

    episode_start_times = np.full(num_envs, time.time())
    episode_frames = np.zeros(num_envs)
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    episode_clipped_rewards = np.zeros(num_envs)
    episode_modified_rewards = np.zeros(num_envs)
    action_amounts = np.zeros((num_envs, agent.action_space,), dtype=np.int32)
    action_distribution_log_names = ["action_" + str(x) for x in range(agent.action_space)]

    env.reset(seed=conf.seed)
    print("randomizing environments ...", end="")
    states = randomize_env(env)
    states = np.array(states).squeeze()
    print(" done")

    prev_time = time.time()
    while (end_episode is None or episode <= end_episode) \
            and (conf.num_frames is None or total_frames < conf.num_frames) \
            and (conf.max_time is None or time.time() < start_time + conf.max_time):

        frame_log = {}

        # select actions using the agents policy on the given states
        if conf.use_exploration:
            actions, beta, log_ratio = agent.select_action(states, action_prob)
        else:
            actions = agent.select_action(states, action_prob)

        actions = actions.cpu().numpy()

        # for logging
        action_amounts[range(num_envs), actions] += 1

        # step the environments async
        env.step_async(actions)

        # train the agent
        if total_frames > conf.start_learning_after and total_frames % train_per == 0:
            for _ in range(train_reps):
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

                # save for logging
                loss_list[train_frames % conf.loss_avg] = loss
                if conf.use_per:
                    weight_list[train_frames % conf.loss_avg] = weights

                train_frames += 1

        # update the target model
        if total_frames > conf.start_learning_after \
                and total_frames % conf.target_model_period == 0 \
                and conf.use_double:
            agent.update_target_model()

        # save the agent
        if total_frames % conf.save_agent_per_frames == 0 and total_frames > 0:
            save_agent(agent, total_frames, conf.log_wandb)

        # get the next states, rewards and dones from the envs
        next_states, rewards, dones, _ = env.step_wait()
        next_states = np.array(next_states).squeeze()

        clipped_rewards = np.clip(rewards, -1, 1)

        episode_rewards += rewards
        episode_clipped_rewards += clipped_rewards

        # we do not clip the total with this softmax exploration
        if conf.use_exploration:
            rewards += rewards - (1 / beta) * log_ratio
            episode_modified_rewards += rewards
        else:
            if conf.clip_reward:
                rewards = clipped_rewards

        # add the transitions to the replay buffer
        agent.add_transitions(states, actions, rewards, dones)

        if dones.any():
            if dones[0]:
                if conf.save_video_per_episodes is not None and video_episode % conf.save_video_per_episodes == 0:
                    frame_log["video"] = wandb.Video(os.path.join(conf.tmp_vid_folder, str(video_episode) + ".mp4"))
                video_episode += 1

            d_sum = dones.sum()
            frame_log["multiple_dones"] = d_sum
            for i, idx in enumerate(np.nonzero(dones)[0]):
                episode += 1
                episode_log = {}
                episode_log["episode_finished"] = episode
                episode_log["episode_reward"] = episode_rewards[idx].item()
                episode_log["episode_clipped_reward"] = episode_clipped_rewards[idx].item()
                if conf.use_exploration:
                    episode_log["episode_modified_reward"] = episode_modified_rewards[idx].item()
                episode_log["episode_length"] = episode_lengths[idx].item()

                action_distribution_dict = dict(
                    zip(action_distribution_log_names, action_amounts[idx] / action_amounts[idx].sum()))
                episode_log.update(action_distribution_dict)

                if not conf.use_noisy:
                    episode_log["episode_exploration_rate"] = agent.epsilon
                if conf.use_per:
                    episode_log["episode_per_beta"] = agent.replay_buffer.beta

                episode_frame = total_frames - (d_sum - i) + 1
                wandb.log(episode_log, step=episode_frame)

                episode_rewards[idx] = 0
                episode_clipped_rewards[idx] = 0
                episode_modified_rewards[idx] = 0
                episode_lengths[idx] = 0
                action_amounts[idx].fill(0)

        # log fps
        if total_frames % (num_envs * 50) == 0 and total_frames > 0:
            t = time.time()
            fps = (num_envs * 50) / max(t - prev_time, 0.00001)
            frame_log["fps"] = fps
            prev_time = t

        states = next_states

        if frame_log:
            wandb.log(frame_log, step=total_frames)
        total_frames += num_envs
        episode_lengths += 1

    end_time = time.time()
    t = end_time - start_time
    hours = int((end_time - start_time) // 3600)
    t -= hours * 3600
    minutes = int(t // 60)
    t -= minutes * 60
    seconds = t
    print("Training finished")
    print("Trained for {} frames in {:02d}:{:02d}:{:02.2f}".format(total_frames, hours, minutes, seconds))


def randomize_env(env, steps=1000):
    assert steps > 0
    for _ in range(steps-1):
        actions = env.action_space.sample()
        env.step(actions)
    actions = env.action_space.sample()
    return env.step(actions)[0]

def init_logging(conf):
    config = {}
    for var_name in dir(conf):
        if var_name.startswith("__"):
            continue
        config[var_name] = getattr(conf, var_name)
    wandb.init(project="pdrl", entity="pdrl", mode=("online" if conf.log_wandb else "offline"),
               config=config)


def save_agent(agent, total_frames, save_to_wandb):
    path = get_agent_save_path(total_frames)
    agent.save(path)
    if save_to_wandb:
        wandb.save(path)


def get_agent_save_path(episode):
    filename = "model_" + str(episode) + ".pt"
    return os.path.join(wandb.run.dir, filename)
