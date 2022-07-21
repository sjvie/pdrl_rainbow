import logging
import time
import numpy as np
import wandb


def train_agent(agent, env, conf):
    total_frames = 0
    train_frames = 0
    reward_list = np.zeros(500, dtype=np.int8)
    loss_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    weight_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    action_amounts = np.zeros((agent.action_space,), dtype=np.int32)
    action_distribution_log_names = ["action_" + str(x) for x in range(agent.action_space)]

    episode = 1
    if conf.num_episodes is not None:
        end_episode = episode + conf.num_episodes
    else:
        end_episode = None

    if conf.log_wandb:
        agent.run.watch(agent.model, log='all', log_freq=conf.model_log_freq)

    logging.info("Starting training")
    start_time = time.time()

    while (end_episode is None or episode <= end_episode) \
            and (conf.num_frames is None or total_frames < conf.num_frames) \
            and (conf.max_time is None or time.time() < start_time + conf.max_time):

        state = env.reset(seed=conf.seed)
        state = process_state(state)

        episode_over = False
        episode_frames = 0
        episode_reward = 0
        episode_loss = 0
        episode_start_time = time.time()

        while not episode_over and episode_frames < conf.max_frames_per_episode and (
                conf.num_frames is None or total_frames < conf.num_frames):
            total_frames += 1
            episode_frames += 1

            action = agent.select_action(state)
            action_amounts[action] += 1

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)

            episode_reward += reward

            if conf.clip_reward:
                reward = np.clip(reward, -1, 1)

            agent.step(state, action, reward, done)

            if total_frames > conf.start_learning_after and total_frames % conf.replay_period == 0:
                loss, weights = agent.train()
                loss = loss.cpu().detach().numpy()
                if conf.use_per:
                    weights = weights.cpu().detach().numpy()

                if conf.log_wandb and train_frames % conf.loss_avg == 0 and train_frames >= conf.loss_avg:
                    agent.run.log({"frame_loss_avg": loss_list.mean(),
                                   "frame_loss_min": loss_list.min(),
                                   "frame_loss_max": loss_list.max()
                                   }, step=total_frames)
                    if conf.use_per:
                        agent.run.log({"buffer_tree_sum": agent.replay_buffer.tree.sum(),
                                       "buffer_tree_min": agent.replay_buffer.tree.min(),
                                       "buffer_max_priority_with_alpha": agent.replay_buffer.max_priority ** agent.replay_buffer.alpha,
                                       "frame_weights_avg": weight_list.mean()
                                       }, step=total_frames)

                # log the loss averaged over loss_avg frames
                loss_list[train_frames % conf.loss_avg] = loss
                if conf.use_per:
                    weight_list[train_frames % conf.loss_avg] = weights

                episode_loss += loss.sum()
                train_frames += 1

            if total_frames > conf.start_learning_after \
                    and total_frames % conf.target_model_period == 0 \
                    and conf.use_double:
                agent.update_target_model()
                logging.debug("Updated target model")

            state = next_state
            episode_over = done

            if total_frames % conf.log_per_frames == 0:
                logging.info(
                    'E/ef/tf: {}/{}/{} | Avg. reward: {:.3f} | Avg loss: {:.3f}'.format(episode, episode_frames,
                                                                                        total_frames,
                                                                                        episode_reward / episode_frames,
                                                                                        episode_loss / episode_frames))
        episode_end_time = time.time()
        fps = episode_frames / max(episode_end_time - episode_start_time, 0.0001)

        if conf.log_wandb:
            agent.run.log({"episode_fps": fps}, step=total_frames)
            agent.run.log({"episode_reward": episode_reward}, step=total_frames)
            if not conf.use_noisy:
                agent.run.log({"episode_exploration_rate": agent.epsilon}, step=total_frames)
            if conf.use_per:
                agent.run.log({"episode_per_beta": agent.replay_buffer.beta}, step=total_frames)
            agent.run.log({"episode_length": episode_frames}, step=total_frames)
            action_distribution_dict = dict(zip(action_distribution_log_names, action_amounts / action_amounts.sum()))
            agent.run.log(action_distribution_dict, step=total_frames)

        reward_list[episode % 500] = episode_reward
        if conf.log_episode_end:
            logging.info(
                '[EPISODE END] E/ef/tf: {}/{}/{} | fps: {:.2f} | Avg. reward: {:.3f} | Avg loss: {:.5f}'.format(episode,
                                                                                                                episode_frames,
                                                                                                                total_frames,
                                                                                                                fps,
                                                                                                                episode_reward,
                                                                                                                episode_loss))
        if episode % 200 == 0:
            if agent.action_space == 6:
                logging.info(
                    '[AVERAGE] | Avg. reward: {:.2f} | Actiondistribution 0: {:.2f} 1: {:.2f} 2: {:.2f} 3: {:.2f} 4:{:.2f} 5:{:.2f}'.format(
                        reward_list.mean(),
                        action_amounts[0] / action_amounts.sum(),
                        action_amounts[1] / action_amounts.sum(),
                        action_amounts[2] / action_amounts.sum(),
                        action_amounts[3] / action_amounts.sum(),
                        action_amounts[4] / action_amounts.sum(),
                        action_amounts[5] / action_amounts.sum())
                )
            if agent.action_space == 3:
                logging.info(
                    '[AVERAGE] | Avg. reward: {:.2f} | Actiondistribution 0: {:.2f} 1: {:.2f} 2: {:.2f}'.format(
                        reward_list.mean(), action_amounts[0],
                        action_amounts[1],
                        action_amounts[2])
                )
            if agent.action_space == 4:
                logging.info(
                    '[AVERAGE] | Avg. reward: {:.2f} | Actiondistribution 0: {:.2f} 1: {:.2f} 2: {:.2f} 3: {:.2f}'.format(
                        reward_list.mean(),
                        action_amounts[0],
                        action_amounts[1],
                        action_amounts[2],
                        action_amounts[3])
                )

        action_amounts.fill(0)
        if episode % conf.save_agent_per_episodes == 0 and episode > 0:
            agent.save(conf.agent_save_path + str(episode),conf.save_buffer)

        episode += 1

    end_time = time.time()
    t = end_time - start_time
    hours = int((end_time - start_time) // 3600)
    t -= hours * 3600
    minutes = int(t // 60)
    t -= minutes * 60
    seconds = t
    logging.info("Training finished")
    logging.info("Trained for {} frames in {:02d}:{:02d}:{:02.2f}".format(total_frames, hours, minutes, seconds))

    agent.save(conf.agent_save_path + str(episode))


def process_state(state):
    return np.array(state).squeeze()
