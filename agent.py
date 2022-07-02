import math
import pickle
import random
from pathlib import Path
from torch.nn import functional as F
import torch

import loss_functions
import wandb

from model import Model
from replay_buffer import PrioritizedBuffer, Buffer
import copy


class Agent:

    def __init__(self, observation_shape, action_space, device, seed, conf):
        self.action_space = action_space
        self.conv_channels = conf.frame_stack
        self.batch_size = conf.batch_size
        self.num_atoms = conf.distributional_atoms
        self.device = device
        self.v_min = conf.distributional_v_min
        self.v_max = conf.distributional_v_max
        self.z_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_support = torch.arange(self.v_min, self.v_max + self.z_delta / 2, self.z_delta, device=self.device)
        self.index_offset = torch.arange(0, self.batch_size, 1 / self.num_atoms,
                                         device=self.device).long() * self.num_atoms
        self.n_step_returns = conf.multi_step_n
        self.discount_factor = conf.discount_factor
        self.discount_factor_k = torch.zeros(self.n_step_returns, dtype=torch.float32, device=self.device)
        self.use_per = conf.use_per
        self.use_noisy = conf.use_noisy
        self.epsilon = conf.epsilon_start
        self.epsilon_end = conf.epsilon_end
        self.epsilon_annealing_steps = conf.epsilon_annealing_steps
        self.delta_eps = (self.epsilon_end - self.epsilon) / self.epsilon_annealing_steps
        self.adam_learning_rate = conf.adam_learning_rate
        self.adam_e = conf.adam_e
        self.replay_buffer_beta_start = conf.replay_buffer_beta_start
        self.replay_buffer_alpha = conf.replay_buffer_alpha
        self.replay_buffer_size = conf.replay_buffer_size
        self.use_distributed = conf.use_distributed

        self.online_model = Model(self.conv_channels, self.action_space, device, conf)
        self.target_model = Model(self.conv_channels, self.action_space, device, conf)
        self.target_model.load_state_dict(self.online_model.state_dict())

        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          lr=self.adam_learning_rate,
                                          eps=self.adam_e)

        self.replay_buffer_prio_offset = conf.replay_buffer_prio_offset
        self.seed = seed

        self.replay_buffer_beta = self.replay_buffer_beta_start

        if self.use_distributed:
            self.get_loss = loss_functions.get_distributional_loss
        else:
            self.get_loss = loss_functions.get_huber_loss

        if self.use_per:
            self.replay_buffer = PrioritizedBuffer(self.replay_buffer_size,
                                                   observation_shape,
                                                   device,
                                                   conf
                                                   )
        else:
            self.replay_buffer = Buffer(self.replay_buffer_size,
                                        observation_shape,
                                        device,
                                        conf
                                        )

        # initializing Loging over Weights and Biases
        if conf.log_wandb:
            self.run = wandb.init(project="pdrl", entity="pdrl",
                                  config={
                                      "config_name": conf.name,
                                      "adam_learning_rate": conf.adam_learning_rate,
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
                                      "seed": self.seed,
                                      "use_per": conf.use_per,
                                      "use_distributed": conf.use_distributed,
                                      "multi_step_n": conf.multi_step_n,
                                      "use_noisy": conf.use_noisy,
                                      "loss_avg": conf.loss_avg,
                                      "start_learning_after": conf.start_learning_after,
                                      "device": self.device,
                                      "env": conf.env_name
                                  })

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def step(self, state, action, reward, done):
        state = torch.Tensor(state).to(self.device)

        self.replay_buffer.add(state, action, reward, done)

    def train(self):
        batch, weights, idxs = self.replay_buffer.get_batch(batch_size=self.batch_size)
        states, actions, rewards, n_next_states, dones = batch

        actions = actions.long()
        dones = dones.long()

        states = states / 255.0
        n_next_states = n_next_states / 255.0

        loss = self.get_loss(self, states, actions, rewards, n_next_states, dones)

        loss = loss.squeeze()
        loss_copy = loss.clone()

        # update the priorities in the replay buffer
        if self.use_per:
            for i in range(self.batch_size):
                # in the PER paper they used a small constant to prevent that the loss is 0
                self.replay_buffer.set_prio(idxs[i].item(), abs(loss[i].item()) + self.replay_buffer_prio_offset)
            loss = loss * weights

        # use the average loss of the batch
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_copy, weights

    def select_action(self, np_state):
        """
        :param np_state: (np array) numpy array with shape observation_shape
        :return (int): index of the selected action
        """

        self.epsilon += self.delta_eps
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

        if random.random() > self.epsilon or self.use_noisy:
            state = torch.from_numpy(np_state).to(self.device)

            state = state / 255.0
            with torch.no_grad():
                if self.use_distributed:

                    # select action using online model
                    Q_dist = self.online_model(state)

                    # get expected Q values
                    Q = (Q_dist * self.z_support).sum(dim=1)

                    # get action_index with maximum Q value
                    action_index = torch.argmax(Q)
                else:
                    Q = self.online_model(state)
                    # get action_index with maximum Q value
                    action_index = torch.argmax(Q, dim=-1)

            # return the index of the action
            return action_index.item()
        else:
            return random.choice(range(0, self.action_space))

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + "/replay.pickle", "wb") as f:
            pickle.dump(self.replay_buffer, f)
        torch.save(self.online_model.state_dict(), path + "/online.pt")
        torch.save(self.target_model.state_dict(), path + "/target.pt")

    def load(self, path):
        with open(path + "/replay.pickle", "rb") as f:
            self.replay_buffer = pickle.load(f)
        self.online_model.load_state_dict(torch.load(path + "/online.pt"))
        self.target_model.load_state_dict(torch.load(path + "/target.pt"))
