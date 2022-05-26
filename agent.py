import torch

from config import Config
from model import Model
from replay_buffer import PrioritizedBuffer
import copy

device = torch.device(Config.device if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, input_dim, action_space, num_atoms, batch_size):
        self.input_dim = input_dim
        self.action_space = action_space
        self.batch_size = batch_size

        self.online_model = Model(input_dim, action_space, num_atoms)
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

        # TODO: calculate loss (or implement in method below)
        # get approximating distribution of the model (target and online model)
        # calculate target distribution (n step returns)
        # project target distribution onto same support as other distribution
        # get Kullbeck-Leibler divergence of target distribution and the approximating distribution

        loss = None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calc_loss(self, state, action, reward, next_state, done):
        pass

    def select_action(self, state):
        return self.online_model.select_action(state)

    # TODO
