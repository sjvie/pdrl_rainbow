import numpy as np

import config
import gym_wrappers
from agent import Agent
from config import Config
import gym


def main():
    # todo: log stuff

    env, input_dim, action_space = cart_pole()

    # env, input_dim, action_space = pong()

    # TODO: what about action repetitions?

    agent = Agent(input_dim,
                  action_space,
                  Config.distributional_atoms,
                  Config.distributional_v_min,
                  Config.distributional_v_max,
                  Config.discount_factor,
                  Config.batch_size,
                  Config.multi_step_n,
                  conv=Config.conv,
                  observation_dt=Config.observation_dt)

    total_frames = 0

    for episode in range(1, Config.num_episodes + 1):
        state = env.reset()
        game_over = False
        episode_frames = 0
        episode_reward = 0
        episode_loss = 0

        while not game_over and episode_frames < Config.max_frames_per_episode:
            total_frames += 1
            episode_frames += 1

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
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

        print('Episode: {} frames: {} Reward: {:.3f} Avg loss: {:.3f}'.format(episode, episode_frames, episode_reward,
                                                                              episode_loss / episode_frames))


def cart_pole():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    action_space = env.action_space.n

    config.set_cart_pole_config()

    return env, input_dim, action_space


def pong():
    env = gym_wrappers.make_atari("ALE/Pong-v5")
    env = gym_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=True)
    """
    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    env = gym.wrappers.ResizeObservation(env, (Config.observation_width, Config.observation_height))
    env = gym.wrappers.FrameStack(env, Config.frame_stack)
    env = gym.wrappers.FlattenObservation(env)
    """
    input_dim = Config.observation_width * Config.observation_height * Config.frame_stack
    action_space = env.action_space.n

    return env, input_dim, action_space


if __name__ == "__main__":
    print("Hello, I am %s!" % Config.name)
    main()
