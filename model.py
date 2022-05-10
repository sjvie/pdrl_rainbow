import math

import cupy as np
import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, input_dim, action_space):
        super().__init__()

        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.value = nn.Sequential(
            #nn.Linear(in_channels= 64, out_channels= 512), nn.ReLU(),   #TODO: NoisyLinear Klasse erstellen
            #nn.NoisyLinear(512,1)
            nn.Linear(in_channels= 64, out_channels= 512), nn.ReLU(),
            nn.Linear(512,1)
        )

        self.advantage = nn.Sequential(
            #nn.NoisyLinear(in_channels= 64, out_channels=512), nn.ReLU(),      #TODO: NoisyLinear Klasse erstellen
            #nn.NoisyLinear(512,action_space)
            nn.Linear(in_channels= 64, out_channels=512), nn.ReLU(),
            nn.Linear(512,action_space)
        )

    def forward(self, input):
        value = self.value(self.conv(input))
        advantage = self.advantage(self.conv(input))
        Q = value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return Q


    def get_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

    # TODO
