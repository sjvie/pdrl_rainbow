import torch

from config import Config
from model import Model
from replay_buffer import PrioritizedBuffer


class Agent:

    def __init__(self, input_dim, action_space):
        self.input_dim = input_dim
        self.action_space = action_space

        self.online_model = Model(input_dim, action_space)
        self.target_model = Model(input_dim, action_space)
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
        pass

    def calc_loss(self, state, action, reward, next_state, done):
        pass

    def select_action(self, state):
        return self.online_model.select_action(state)

    # TODO
