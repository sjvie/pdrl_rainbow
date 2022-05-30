import numpy as np
import torch

from agent import Agent
from config import Config
import gym


def main():
    # todo: log stuff

    env = gym.make("CartPole-v1")
    # TODO: is this needed? implement differently? what about action repetitions?
    # env = gym.wrappers.FrameStack(env, Config.frame_stack)

    input_dim = env.observation_space.shape[0]
    #action_space = env.action_space.n
    action_space = 2

    agent = Agent(input_dim,
                  action_space,
                  Config.distributional_atoms,
                  Config.distributional_v_min,
                  Config.distributional_v_max,
                  Config.discount_factor,
                  Config.batch_size,
                  conv=False)

    total_frames = 0

    for episode in range(1, Config.num_episodes + 1):
        state = env.reset()
        game_over = False
        episode_frames = 0

        while not game_over and episode_frames < Config.max_frames_per_episode:
            total_frames += 1
            episode_frames += 1

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = np.clip(reward, -1, 1)
            # todo: action repetitions?
            agent.step(state, action, reward, next_state, done)

            # TODO: total_frames or episode_frames?
            if total_frames > Config.start_learning_after and total_frames % Config.replay_period == 0:
                if total_frames % Config.target_model_period == 0:
                    agent.update_target_model()
                agent.train()

            state = next_state
            game_over = done



if __name__ == "__main__":
    print("Hello, I am %s!" % Config.name)
    main()
