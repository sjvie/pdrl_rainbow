import os
import time
import numpy as np

import env_utils
import util
import wandb


def train_agent(agent, env, conf):
    num_envs = conf.num_envs
    # these values need to be a multiple of the number of parallel envs to work correctly
    assert conf.target_model_period % num_envs == 0
    assert conf.sample_repetitions % num_envs == 0 or num_envs % conf.sample_repetitions == 0
    assert conf.batch_size % num_envs == 0 or num_envs % conf.batch_size == 0
    assert conf.batch_size % conf.sample_repetitions == 0 or conf.sample_repetitions % conf.batch_size == 0

    if num_envs * conf.sample_repetitions > conf.batch_size:
        train_reps_per_step = (num_envs * conf.sample_repetitions) // conf.batch_size
        train_per_steps = 1
    else:
        train_reps_per_step = 1
        train_per_steps = (conf.batch_size // conf.sample_repetitions) // num_envs

    conf.train_reps_per_step = train_reps_per_step
    conf.train_per_steps = train_per_steps

    total_frames = 0
    train_steps = 0
    steps = 0
    episode = 1
    video_episode = 1
    next_save_agent_frame = conf.save_agent_per_frames

    # if not conf.terminal_on_life_loss:
    #    lives = np.zeros(conf.num_envs, dtype=np.uint8)

    # for logging
    loss_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    weight_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)

    get_epsilon = util.LinearValue(conf.epsilon_start, conf.epsilon_end, 0, conf.epsilon_annealing_steps)
    get_beta = util.LinearValue(conf.replay_buffer_beta_start, conf.replay_buffer_beta_end, 0,
                                conf.replay_buffer_beta_annealing_steps)

    util.init_logging(conf)
    wandb.watch(agent.model, log='all', log_freq=conf.model_log_freq)

    print("Starting training")
    start_time = time.time()
    # uniform action distribution
    action_prob = np.ones(agent.action_space) * (1 / agent.action_space)
    distribution = np.empty(agent.action_space)
    beta = 0

    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    episode_clipped_rewards = np.zeros(num_envs)
    episode_modified_rewards = np.zeros(num_envs)
    action_amounts = np.zeros((num_envs, agent.action_space,), dtype=np.int32)
    action_distribution_log_names = ["action_" + str(x) for x in range(agent.action_space)]

    states = env.reset(seed=conf.seed)
    if conf.randomize_env_steps > 0:
        print("randomizing environments ...", end="")
        states = env_utils.randomize_env(env, conf.randomize_env_steps)
        print(" done")
    states = np.array(states).squeeze()

    prev_time = time.time()
    while conf.num_frames == -1 or total_frames < conf.num_frames:
        frame_log = {}

        agent.epsilon = get_epsilon(total_frames)
        if conf.use_per:
            agent.replay_buffer.beta = get_beta(total_frames)

        # select actions using the agent's policy on the given states
        if conf.use_exploration:
            actions, beta, log_ratio = agent.select_action(states, action_prob)
        else:
            actions = agent.select_action(states, action_prob)

        # for logging
        action_amounts[range(num_envs), actions] += 1

        # step the environments async
        env.step_async(actions)

        # train the agent
        if total_frames > conf.start_learning_after and steps % train_per_steps == 0:
            for _ in range(train_reps_per_step):
                loss, weights = agent.train()
                loss = loss.cpu().detach().numpy()
                if conf.use_per:
                    weights = weights.cpu().detach().numpy()

                if train_steps % conf.loss_avg == 0 and train_steps >= conf.loss_avg:
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
                loss_list[train_steps % conf.loss_avg] = loss
                if conf.use_per:
                    weight_list[train_steps % conf.loss_avg] = weights

                train_steps += 1

        # update the target model
        if total_frames > conf.start_learning_after \
                and total_frames % conf.target_model_period == 0 \
                and conf.use_double:
            agent.update_target_model()

        # save the agent
        if total_frames >= next_save_agent_frame:
            util.save_agent(agent, total_frames, conf.log_wandb)
            next_save_agent_frame += conf.save_agent_per_frames

        # get the next states, rewards and dones from the envs
        next_states, rewards, dones, infos = env.step_wait()
        next_states = np.array(next_states).squeeze()

        # end episode after life loss when the env does not reset on life loss
        # as everyone else probably uses the score of the full game -> don't do this
        # if not conf.terminal_on_life_loss:
        #    info_lives = np.array([i["lives"] for i in infos])
        #    dones = np.logical_or(dones, np.logical_and(info_lives < lives, info_lives > 0))
        #    lives = info_lives

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
            if dones[0] and "video_path" in infos[0]:
                frame_log["video"] = wandb.Video(infos[0]["video_path"])
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
        steps += 1
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
