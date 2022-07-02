import logging
import time
import numpy as np
import wandb


def train_agent(agent, env, conf):
    total_frames = 0
    train_frames = 0
    action_list = np.zeros(agent.action_space)
    reward_list = np.zeros(500, dtype=np.int8)
    loss_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)
    weight_list = np.zeros((conf.loss_avg, conf.batch_size), dtype=np.float32)

    episode = 1
    if conf.num_episodes is not None:
        end_episode = episode + conf.num_episodes
    else:
        end_episode = None
    logging.info("Starting training")
    start_time = time.time()

    if conf.log_wandb:
        agent.run.watch(agent.online_model, log='all', log_freq=conf.model_log_freq)

    while (end_episode is None or episode <= end_episode) \
            and (conf.num_frames is None or total_frames < conf.num_frames)\
            and (conf.max_time is None or time.time() < start_time + conf.max_time):

        state = env.reset()
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
            action_list[action] += 1

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)

            if conf.clip_reward:
                reward = np.clip(reward, -1, 1)

            episode_reward += reward
            # todo: action repetitions?
            agent.step(state, action, reward, done)

            if total_frames > conf.start_learning_after and total_frames % conf.replay_period == 0:
                loss, weights = agent.train()
                loss = loss.cpu().detach().numpy()
                #loss = loss.mean()
                if conf.use_per:
                    weights = weights.cpu().detach().numpy()
                #    weights = weights.mean()

                if conf.log_wandb and train_frames % conf.loss_avg == 0 and train_frames >= conf.loss_avg:
                    agent.run.log({"frame_loss_avg": loss_list.mean()}, step=total_frames)
                    agent.run.log({"frame_loss_min": loss_list.min()}, step=total_frames)
                    agent.run.log({"frame_loss_max": loss_list.max()}, step=total_frames)
                    # todo TMP
                    if conf.use_per:
                        agent.run.log({"frame_weights_avg": weight_list.mean()}, step=total_frames)
                        agent.run.log({"buffer_tree_sum": agent.replay_buffer.tree.sum()}, step=total_frames)
                        agent.run.log({"buffer_tree_min": agent.replay_buffer.tree.min()}, step=total_frames)
                        agent.run.log({"buffer_tree_max": agent.replay_buffer.tree.max()}, step=total_frames)
                        #agent.run.log({"buffer_tree": agent.replay_buffer.tree.sum_array[agent.replay_buffer.tree.data_index_offset:]}, step=total_frames)

                # log the loss averaged over loss_avg frames
                loss_list[train_frames % conf.loss_avg] = loss
                weight_list[train_frames % conf.loss_avg] = weights

                #agent.run.log({"mean_loss_over_time": loss.item()})
                episode_loss += loss.sum()
                train_frames += 1

            if total_frames > conf.start_learning_after and total_frames % conf.target_model_period == 0:
                agent.update_target_model()
                logging.debug("Updated target model")

            """column=[i for i in range(agent.action_space)]
            data = [action_list]
            table=wandb.Table(data=data,columns=column)
            agent.run.log({"actions": table})
            """

            state = next_state
            episode_over = done

            if total_frames % conf.log_per_frames == 0:
                logging.info(
                    'E/ef/tf: {}/{}/{} | Avg. reward: {:.3f} | Avg loss: {:.3f}'.format(episode, episode_frames,
                                                                                        total_frames,
                                                                                        episode_reward / episode_frames,
                                                                                        episode_loss / episode_frames))
        episode_end_time = time.time()
        fps = episode_frames / (episode_end_time - episode_start_time)

        if conf.log_wandb:
            agent.run.log({"episode_fps": fps}, step=total_frames)
            agent.run.log({"episode_reward": episode_reward}, step=total_frames)
            if not conf.use_noisy:
                agent.run.log({"episode_exploration_rate": agent.epsilon}, step=total_frames)
            agent.run.log({"episode_length": episode_frames}, step=total_frames)

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
                        action_list[0] / action_list.sum(),
                        action_list[1] / action_list.sum(),
                        action_list[2] / action_list.sum(),
                        action_list[3] / action_list.sum(),
                        action_list[4] / action_list.sum(),
                        action_list[5] / action_list.sum())
                )
            if agent.action_space == 3:
                logging.info(
                    '[AVERAGE] | Avg. reward: {:.2f} | Actiondistribution 0: {:.2f} 1: {:.2f} 2: {:.2f}'.format(
                        reward_list.mean(), action_list[0],
                        action_list[1],
                        action_list[2])
                )
            if agent.action_space == 4:
                logging.info(
                    '[AVERAGE] | Avg. reward: {:.2f} | Actiondistribution 0: {:.2f} 1: {:.2f} 2: {:.2f} 3: {:.2f}'.format(
                        reward_list.mean(),
                        action_list[0],
                        action_list[1],
                        action_list[2],
                        action_list[3])
                )
            action_list = np.zeros(agent.action_space)

        if episode % conf.save_agent_per_episodes == 0 and episode > 0:
            agent.save(conf.agent_save_path + str(episode))

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
