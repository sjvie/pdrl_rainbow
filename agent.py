import math
import pickle
import random
from pathlib import Path
from torch.nn import functional as F
import torch
import wandb

#from test_config import Config
from config import Config
from model import Model
from replay_buffer import PrioritizedBuffer
import copy

device = torch.device(Config.device if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, observation_shape, conv_channels, action_space, num_atoms, v_min, v_max, discount_factor,
                 batch_size, n_step_returns, tensor_replay_buffer, use_per, use_multistep, noisy, epsilon, epsilon_min, distributed):
        self.action_space = action_space
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_delta = (v_max - v_min) / (num_atoms - 1)
        self.z_support = torch.arange(self.v_min, self.v_max + self.z_delta / 2, self.z_delta, device=device)
        self.index_offset = torch.arange(0, self.batch_size, 1 / self.num_atoms, device=device).long() * self.num_atoms
        self.n_step_returns = n_step_returns
        self.discount_factor = discount_factor
        self.discount_factor_k = torch.zeros(self.n_step_returns, dtype=torch.float32, device=device)
        self.use_per = use_per
        self.use_multistep = use_multistep
        self.noisy = noisy
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.annealing_steps=1000000
        #epsilon annealing over 1M steps to a minimum of 0.01
        self.delta_eps = (self.epsilon-self.epsilon_min)/self.annealing_steps
        self.distributed = distributed
        self.online_model = Model(conv_channels, action_space, num_atoms,noisy=self.noisy,distributed=self.distributed,device=device)
        self.target_model = copy.deepcopy(self.online_model)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          lr=Config.adam_learning_rate,
                                          eps=Config.adam_e)

        # todo: linearly increase beta up to Config.replay_buffer_end
        self.replay_buffer_beta = Config.replay_buffer_beta_start
        self.tensor_replay_buffer = tensor_replay_buffer
        self.replay_buffer = PrioritizedBuffer(Config.replay_buffer_size,
                                               observation_shape,
                                               n_step_returns,
                                               Config.replay_buffer_alpha,
                                               self.replay_buffer_beta,
                                               device=device,
                                               discount_factor=self.discount_factor,
                                               tensor_memory=tensor_replay_buffer,
                                               per=self.use_per,
                                               multistep=self.use_multistep
                                               )

        self.episode_counter = 0
        self.training_counter = 0

        # initializing Loging over Weights and Biases
        self.run = wandb.init(project="pdrl", entity="gdlktemo")
        wandb.config ={
            "learning_rate" : Config.adam_learning_rate,
            "max_episodes": Config.num_episodes,
            "discount_factor": Config.discount_factor,
            "noisy_net_sigma": Config.noisy_sigma_zero,
            "multistep n": Config.multi_step_n
        }

    def next_episode(self):
        self.episode_counter += 1

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def step(self, state, action, reward, done):
        if self.tensor_replay_buffer:
            state = torch.Tensor(state).to(device)

        self.replay_buffer.add(state, action, reward, done)

    def train(self):
        batch, weights, idxs = self.replay_buffer.get_batch(batch_size=self.batch_size)
        states, actions, rewards, n_next_states, dones = batch

        if not self.tensor_replay_buffer:
            # convert to tensors
            states = torch.Tensor(states).to(device)
            actions = torch.Tensor(actions).to(device).long()
            rewards = torch.Tensor(rewards).to(device)
            n_next_states = torch.Tensor(n_next_states).to(device)
            dones = torch.Tensor(dones).to(device).long()
            weights = torch.Tensor(weights).to(device)
        else:
            actions = actions.long()
            dones = dones.long()

        states = states / 255.0
        n_next_states = n_next_states / 255.0
        if self.distributed:
            # initialize target distribution matrix
            m = torch.zeros(self.batch_size, self.num_atoms).to(device)

            # logarithmic output of online model for states
            # shape (batch_size, action_space, num_atoms)
            log_q_dist = self.online_model.forward(states, log=True)
            log_q_dist_a = log_q_dist[range(self.batch_size), actions]

            with torch.no_grad():
                # non-logarithmic output of online model for n next states
                q_online = self.online_model(n_next_states)

                # get best actions for next states according to online model
                # a* = argmax_a(sum_i(z_i *p_i(x_{t+1},a)))
                a_star = torch.argmax((q_online * self.z_support).sum(-1), dim=1)

                # todo: assert shape of log_q_dist / q_dist

                # Double DQN part
                # non-logarithmic output of target model for n next states
                q_target = self.target_model.forward(n_next_states)

                # get distributions for action a* selected by online model
                next_dist = q_target[range(self.batch_size), a_star]

                # calculate n step return
                # G = torch.zeros(self.batch_size, dtype=torch.float32).to(device)
                # for i in range(self.batch_size):
                #     for j in range(self.n_step_returns):
                #         G[i] += rewards[i][j] * self.discount_factor ** j

                #G = torch.sum(rewards * self.discount_factor_k, -1)

                # Tz = r + gamma*(1-done)*z
                T_z = rewards.unsqueeze(-1) + torch.outer(self.discount_factor ** self.n_step_returns * (1 - dones),
                                                    self.z_support)

                # eingrenzen der Werte
                T_z = T_z.clamp(min=self.v_min, max=self.v_max)

                # bj ist hier der index der atome auf denen die Distribution liegt
                bj = (T_z - self.v_min) / self.z_delta

                # l und u sind die ganzzahligen indizes auf die bj projeziert werden soll
                l = bj.floor().long()
                u = bj.ceil().long()

                # values to be added at the l and u indices
                l_add = (u - bj) * next_dist
                u_add = (bj - l) * next_dist

                # values to be added at the indices where l == u == bj
                # todo: is this needed? It does not seem to be a part of the algorithm in the dist paper
                same_add = (u == l) * next_dist

                # add values to m at the given indices
                m.view(-1).index_add_(0, u.view(-1) + self.index_offset, u_add.view(-1))
                m.view(-1).index_add_(0, l.view(-1) + self.index_offset, l_add.view(-1))
                m.view(-1).index_add_(0, l.view(-1) + self.index_offset, same_add.view(-1))

                # Die Wahrscheinlichkeiten, die eigentlich auf den index bj fallen würden werden jetzt auf die indexe l und u verteilt
                # offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                #                          ).long()
                #           .unsqueeze(1)
                #           .expand(self.batch_size, self.num_atoms)
                #           .to(device)
                #           )

            # get Kullbeck-Leibler divergence of target and approximating distribution
            # the KL divergence calculation has some issues as parts of m can be 0.
            # this makes the log(m) = -inf and loss = nan
            # loss = torch.sum(m * torch.log(m) - m * log_q_dist, dim=-1) # KL divergence
            loss = - torch.sum(m * log_q_dist_a, dim=-1)  # cross entropy
        #Assuming if we don't choose Distributed RL we use Duelling RL
        #TODO N-step
        else:
            with torch.no_grad():
                rewards = rewards.unsqueeze(-1)
                #compute next Q-value using target_network
                next_q_values = self.target_model(n_next_states)
                #take action with highest q_value, _ gets the indices of the max value
                next_q_values,_ = next_q_values.max(dim=1)
                #avoid broadcast issue /just to be sure
                next_q_values = next_q_values.reshape(-1,1)
                target_q_values = rewards + (1 - dones.unsqueeze(1)) * self.discount_factor * next_q_values

            current_q_values = self.online_model(states)
            #ToDO: why does this work?!?
            current_q_values = current_q_values.gather(1,actions.unsqueeze(-1))
            #use Huberloss for error clipping, prevents exploding gradients

        loss = F.mse_loss(current_q_values,target_q_values,reduction='none')

        # update the priorities in the replay buffer
        if self.use_per:
            for i in range(self.batch_size):
                self.replay_buffer.set_prio(idxs[i].item(), loss[i].item())
                self.run.log({"priorities":loss[i].item()})
        #TODO: wird der loss mit dem weight multipliziert, nachdem die replaybuffer experiences geupdatet werden?
        # weight loss using weights from the replay buffer
        loss = loss * weights

        # use the average loss of the batch
        loss = loss.mean()
        self.run.log({"mean_loss_over_time":loss})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_counter += 1
        return loss

    def select_action(self, np_state):
        """
        :param np_state: (np array) numpy array with shape observation_shape
        :return (int): index of the selected action
        """
        self.epsilon -= self.delta_eps
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        if(random.random()>self.epsilon or self.noisy):
            state = torch.from_numpy(np_state).to(device)

            state = state / 255.0
            with torch.no_grad():
                if self.distributed:

                    # select action using online model
                    Q_dist = self.online_model(state)

                    # get expected Q values
                    Q_dist = Q_dist * self.z_support
                    Q = Q_dist.sum(dim=1)

                    # get action_index with maximum Q value
                    action_index = torch.argmax(Q)
                else:
                    Q = self.online_model(state)
                    #get action_index with maximum Q value
                    action_index = torch.argmax(Q,dim=1)

            # return the index of the action
            return action_index.item()
        else:
            return random.choice(range(0,self.action_space))

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
