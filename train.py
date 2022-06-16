import logging
import time

import config

import numpy as np
import wandb

def train_agent(agent, env, conf=config.Config):

    if conf.num_episodes is None and conf.num_frames is None and conf.max_time is None:
        raise ValueError("Either num_episodes, num_frames or max_time must be specified")

    total_frames = 0

    episode = agent.episode_counter + 1
    if conf.num_episodes is not None:
        end_episode = episode + conf.num_episodes
    else:
        end_episode = None
    logging.info("Starting training")
    start_time = time.time()
    agent.run.watch(agent.online_model,log='all')
    """while (end_episode is None or episode <= end_episode) \
            and (conf.num_frames is None or total_frames < conf.num_frames)\
            and (conf.max_time is None or time.time() < start_time + conf.max_time):"""
    while(True):

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

            agent.run.log({"action": action})

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            reward = np.clip(reward, -1, 1)
            episode_reward += reward
            # todo: action repetitions?
            agent.step(state, action, reward, done)

            if total_frames > conf.start_learning_after and total_frames % conf.replay_period == 0:
                loss = agent.train()
                episode_loss += loss

            if total_frames > conf.start_learning_after and total_frames % conf.target_model_period == 0:
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
        agent.run.log({"episode_reward":episode_reward})
        episode_end_time = time.time()
        if conf.log_episode_end:
            fps = episode_frames / (episode_end_time - episode_start_time)
            logging.info(
                '[EPISODE END] E/ef/tf: {}/{}/{} | fps: {:.2f} | Avg. reward: {:.3f} | Avg loss: {:.3f}'.format(episode,
                                                                                                                episode_frames,
                                                                                                                total_frames,
                                                                                                                fps,
                                                                                                                episode_reward / episode_frames,
                                                                                                                episode_loss / episode_frames))

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
