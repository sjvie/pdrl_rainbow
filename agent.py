import math
import pickle
import random
from pathlib import Path
from torch.nn import functional as F
import torch
import numpy as np
import loss_functions
import wandb

from model import Model
from replay_buffer import PrioritizedBuffer, Buffer
import copy


class Agent:

    def __init__(self, observation_shape, action_space, device, conf):
        self.action_space = action_space
        self.conv_channels = conf.frame_stack
        self.input_features = observation_shape[0]

        self.batch_size = conf.batch_size
        self.device = device
        self.use_exploration = conf.use_exploration
        self.use_distributional = conf.use_distributional
        self.num_atoms = conf.distributional_atoms
        self.v_min = conf.distributional_v_min
        self.v_max = conf.distributional_v_max
        self.z_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_support = torch.arange(self.v_min, self.v_max + self.z_delta / 2, self.z_delta, device=self.device)
        self.index_offset = torch.arange(0, self.batch_size, 1 / self.num_atoms,
                                         device=self.device).long() * self.num_atoms

        if self.use_distributional:
            self.get_loss = loss_functions.get_distributional_loss
        else:
            self.get_loss = loss_functions.get_huber_loss
            self.num_atoms = 1
            self.z_support = torch.tensor([1], device=self.device)

        self.n_step_returns = conf.multi_step_n
        self.discount_factor = conf.discount_factor

        self.use_noisy = conf.use_noisy
        self.epsilon = conf.epsilon_start
        self.epsilon_end = conf.epsilon_end
        self.epsilon_annealing_steps = conf.epsilon_annealing_steps
        self.delta_eps = (self.epsilon_end - self.epsilon) / self.epsilon_annealing_steps
        self.exp_beta = conf.exp_beta_start
        self.exp_beta_mid = conf.exp_beta_mid
        self.exp_beta_end = conf.exp_beta_end
        self.exp_beta_annealing_steps = conf.exp_beta_annealing_steps
        self.exp_beta_annealing_steps2 = conf.exp_beta_annealing_steps
        self.delta_exp_beta = (self.exp_beta_mid - self.exp_beta) / self.exp_beta_annealing_steps
        self.delta_exp_beta2 = (self.exp_beta_end - self.exp_beta_mid) / self.exp_beta_annealing_steps

        self.adam_learning_rate = conf.adam_learning_rate
        self.adam_e = conf.adam_e

        self.use_per = conf.use_per
        self.replay_buffer_beta = conf.replay_buffer_beta_start
        self.replay_buffer_alpha = conf.replay_buffer_alpha
        self.replay_buffer_size = conf.replay_buffer_size
        self.replay_buffer_prio_offset = conf.replay_buffer_prio_offset

        self.use_kl_loss = conf.use_kl_loss

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

        self.use_double = conf.use_double

        self.model = Model(action_space=self.action_space, device=device, conf=conf,
                           conv_channels=self.conv_channels, input_features=self.input_features)

        if self.use_double:
            self.target_model = Model(action_space=self.action_space, device=device, conf=conf,
                                      conv_channels=self.conv_channels, input_features=self.input_features)
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.adam_learning_rate,
                                          eps=self.adam_e)

        # initializing Logging over Weights and Biases
        if conf.log_wandb:
            self.run = wandb.init(project="pdrl", entity="pdrl",
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
                                      "device": self.device,
                                      "env": conf.env_name,
                                      "target_model_period": conf.target_model_period,
                                      "cuda_deterministic": conf.cuda_deterministic,
                                      "use_kl_loss": conf.use_kl_loss,
                                      "use_dueling": conf.use_dueling,
                                      "use_double": conf.use_double,
                                  })

    def update_target_model(self):
        assert self.use_double
        self.target_model.load_state_dict(self.model.state_dict())

    def step(self, state, action, reward, done):
        self.replay_buffer.add(state, action, reward, done)

    def train(self):
        batch, weights, idxs = self.replay_buffer.get_batch(batch_size=self.batch_size)
        states, actions, rewards, n_next_states, dones = batch

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        n_next_states = torch.from_numpy(n_next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        idxs = torch.from_numpy(idxs).to(self.device)

        actions = actions.long()
        dones = dones.long()
        idxs = idxs.long()

        if states.dtype == torch.uint8:
            states = states / 255.0
            n_next_states = n_next_states / 255.0

        loss, priorities = self.get_loss(self, states, actions, rewards, n_next_states, dones)

        loss = loss.squeeze()
        loss_copy = loss.clone()

        # update the priorities in the replay buffer
        if self.use_per:
            weights = torch.from_numpy(weights).to(self.device)
            for i in range(self.batch_size):
                # in the PER paper they used a small constant to prevent that the loss is 0
                self.replay_buffer.set_prio(idxs[i].item(), priorities[i].item())
            loss = loss * weights

        # use the average loss of the batch
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_copy, weights

    def select_action(self, np_state, action_prob):
        """
        :param action_prob: tensor with probability distribution (in our case uniform distribution)
        :param np_state: (np array) numpy array with shape observation_shape
        :return (int): index of the selected action
        """

        self.epsilon += self.delta_eps
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end
        if self.exp_beta < self.exp_beta_mid:
            self.exp_beta += self.delta_exp_beta
        elif self.exp_beta >= self.exp_beta_mid:
            self.exp_beta += self.delta_exp_beta2
        if self.exp_beta > self.exp_beta_end:
            self.exp_beta = self.exp_beta_end
        state = torch.from_numpy(np_state).to(self.device)

        if state.dtype == torch.uint8:
            state = state / 255.0

        if not self.use_exploration and (random.random() > self.epsilon or self.use_noisy):

            with torch.no_grad():
                q_dist = self.model(state, log=False)
                q = (q_dist * self.z_support).sum(dim=-1)

                action_index = torch.argmax(q, dim=-1)

            # return the index of the action
            return action_index.item()

        elif self.use_exploration:
            action_q_values = self.model(state, log=False).squeeze()
            # since we use the softmax for these values, we can "normalize" them by subtracting the maximum
            # from each value as it still preserves the order of magnitude
            # this also prevents possible overflows (since the softmax function uses the e-function)
            max_action_prob = torch.max(action_q_values).item()
            action_probabilities = torch.add(action_q_values, -max_action_prob)

            # we can clip the lower bound for action values, since they (most-likely) are not considered anyway
            # also used in https://openreview.net/pdf?id=HyEtjoCqFX
            action_probabilities = torch.clip(action_probabilities, min=-10, max=None)

            # we sample our actions after the softmax policy :
            # pi*(a|s) = exp(exp_beta * Q(a,s)) / sum_a' exp(exp_beta*Q(a',s)
            distribution = softmax(action_probabilities, self.exp_beta)
            action = torch.multinomial(distribution, 1)
            action = action[0].item()
            log_ratio = ((distribution[action]/action_prob[action]).log()).sum().item()
            return action, self.exp_beta, log_ratio
        else:
            return random.choice(range(0, self.action_space))

    def save(self, path,save_buffer=False):
        if save_buffer:
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(path + "/replay.pickle", "wb") as f:
                pickle.dump(self.replay_buffer, f)
            torch.save(self.model.state_dict(), path + "/online.pt")
            if self.use_double:
                torch.save(self.target_model.state_dict(), path + "/target.pt")

    def load(self, path):
        with open(path + "/replay.pickle", "rb") as f:
            self.replay_buffer = pickle.load(f)
        self.model.load_state_dict(torch.load(path + "/online.pt"))
        if self.use_double:
            self.target_model.load_state_dict(torch.load(path + "/target.pt"))


def softmax(action_prob, beta):
    """
        Function to calc Softmax/Boltzmann-Distribution
    """
    exp = torch.exp(beta*action_prob)
    smax = exp/torch.sum(exp)
    #assert torch.sum(smax) == 1.0
    return smax
