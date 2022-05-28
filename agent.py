import torch
from config import Config
from model import Model
from replay_buffer import PrioritizedBuffer
import copy

device = torch.device(Config.device if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, input_dim, action_space, num_atoms, v_min, v_max, discount_factor, batch_size, conv=True):
        self.input_dim = input_dim
        self.action_space = action_space
        self.conv = conv
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_delta = (v_max - v_min) / (num_atoms - 1)
        self.z_support = torch.arange(self.v_min, self.v_max + self.z_delta / 2, self.z_delta)

        self.discount_factor = discount_factor

        self.online_model = Model(input_dim, action_space, num_atoms, conv=conv)
        self.target_model = copy.deepcopy(self.online_model)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          lr=Config.adam_learning_rate,
                                          eps=Config.adam_e)

        # todo: linearly increase beta up to Config.replay_buffer_end
        self.replay_buffer_beta = Config.replay_buffer_beta_start
        self.replay_buffer = PrioritizedBuffer(Config.replay_buffer_size,
                                               input_dim,
                                               Config.replay_buffer_alpha,
                                               self.replay_buffer_beta)

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    # todo: rename method?
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        batch, weights, idxs = self.replay_buffer.get_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        # convert to tensors
        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        next_states = torch.Tensor(next_states).to(device)
        dones = torch.Tensor(dones).to(device)

        # TODO: calculate loss (or implement in method below)
        m = torch.zeros(self.batch_size, self.num_atoms).to(device)
        log_q_dist = self.online_model.forward(states, log=True)  # should be shape(32,51,num_actions)
        q_online = self.online_model(next_states)
        a_star = torch.argmax((q_online * self.z_support).sum(-1), dim=1)  # a* = argmax_a(sum_i(z_i *p_i(x_{t+1},a)))
        # todo: ich weiß nicht in welche Dimension wir das letztendlich summieren müssen, also nehm ich einfach mal die letzte, ev ändern
        # todo: assert shape of log_q_dist /q_dist
        next_dist = self.target_model.forward(next_states)  # Double DQN part
        next_dist = next_dist[self.batch_size, a_star]

        T_zj = rewards + self.discount_factor * (
                    1 - dones) * self.z_support  # Tz = r + gamma*(1-done)*z_j TODO: hier ansetzen für Multistep?
        T_zj = torch.clamp(T_zj, min=self.v_min, max=self.v_max)  # eingrenzen der Werte
        # bj ist hier der index der atome auf denen die Distribution liegt
        bj = (T_zj - self.v_min) / self.z_delta
        # l und u sind die benachbarten indexe auf die bj projeziert werden soll
        l = bj.floor()
        u = bj.ceil()
        # Die Wahrscheinlichkeiten, die eigentlich auf den index bj fallen würden werden jetzt auf die indexe l und u verteilt
        offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                                 ).long()
                  .unsqueeze(1)
                  .expand(self.batch_size, self.num_atoms)
                  .to(device)
                  )

        # get approximating distribution of the model (target and online model)
        # calculate target distribution (n step returns)
        # project target distribution onto same support as other distribution
        # get Kullbeck-Leibler divergence of target distribution and the approximating distribution

        # get Kullbeck-Leibler divergence of target and approximating distribution
        loss = torch.sum(m * torch.log(m) - m * log_q_dist)
        # loss = - torch.sum(m * log_q_dist) # cross entropy

        # update the priorities in the replay buffer
        for i in range(self.batch_size):
            self.replay_buffer.set_prio(idxs[i], loss[i])

        # weight loss using weights from the replay buffer
        loss = loss * weights

        # use the average loss of the batch
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calc_loss(self, state, action, reward, next_state, done):
        pass

    def select_action(self, np_state):
        """
        :param np_state (np array): numpy array with dim [input_dim]
        :return (int): index of the selected action
        """
        state = torch.from_numpy(np_state).to(device)

        with torch.no_grad():
            # select action using online model
            Q_dist = self.online_model(state)

            # get expected Q values
            Q_dist = Q_dist * self.z_support
            Q = Q_dist.sum(dim=1)

            # get action_index with maximum Q value
            action_index = torch.argmax(Q)

        # return the index of the action
        return action_index.item()
