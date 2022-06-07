import numpy as np
import torch

import config
import gym_wrappers
from agent import Agent
from config import Config
import gym


def main():
    # todo: log stuff

    env, observation_shape, action_space = pong()

    # TODO: what about action repetitions?

    print("Cuda available: %s" % torch.cuda.is_available())

    agent = Agent(observation_shape,
                  Config.frame_stack,
                  action_space,
                  Config.distributional_atoms,
                  Config.distributional_v_min,
                  Config.distributional_v_max,
                  Config.discount_factor,
                  Config.batch_size,
                  Config.multi_step_n,
                  observation_dt=Config.observation_dt)

    total_frames = 0

    for episode in range(1, Config.num_episodes + 1):
        state = env.reset()
        state = process_state(state)
        game_over = False
        episode_frames = 0
        episode_reward = 0
        episode_loss = 0

        while not game_over and episode_frames < Config.max_frames_per_episode:
            total_frames += 1
            episode_frames += 1

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            reward = np.clip(reward, -1, 1)
            episode_reward += reward
            # todo: action repetitions?
            agent.step(state, action, reward, done)

            # TODO: total_frames or episode_frames?
            if total_frames > Config.start_learning_after and total_frames % Config.replay_period == 0:
                if total_frames % Config.target_model_period == 0:
                    agent.update_target_model()
                loss = agent.train()
                episode_loss += loss

            state = next_state
            game_over = done

            if total_frames % Config.log_per_frames == 0:
                #print(agent.online_model.conv.state_dict()["0.weight"])
                print('E/ef/tf: {}/{}/{} | Avg. reward: {:.3f} | Avg loss: {:.3f}'.format(episode, episode_frames, total_frames,
                                                                                      episode_reward / episode_frames,
                                                                                      episode_loss / episode_frames))


        if Config.log_episode_end:
            print('[EPISODE END] E/ef/tf: {}/{}/{} | Avg. reward: {:.3f} | Avg loss: {:.3f}'.format(episode, episode_frames,
                                                                                     total_frames,
                                                                                     episode_reward / episode_frames,
                                                                                     episode_loss / episode_frames))


def process_state(state):
    return np.array(state).squeeze()


def cart_pole():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    action_space = env.action_space.n

    config.set_cart_pole_config()

    return env, input_dim, action_space


def pong():
    # env = gym_wrappers.make_atari("ALE/Pong-v5")
    # env = gym_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=True)

    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    env = gym.wrappers.RecordVideo(env, "vid", episode_trigger=lambda x: x % 10 == 0)

    observation_shape = (Config.frame_stack, Config.observation_width, Config.observation_height)
    action_space = env.action_space.n

    return env, observation_shape, action_space


if __name__ == "__main__":
    print("Hello, I am %s!" % Config.name)
    main()
