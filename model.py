import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_dim, action_space):
        super().__init__()

        self.input_dim = input_dim
        self.conv = nn.Sequential(
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )
        self.value = nn.Sequential(
            nn.Linear(in_channels= 64, out_channels= 512)
        )

        self.advantage = nn.Sequential(
            nn.Linear(in_channels= 64, out_channels=512)

        )

        # TODO

    # TODO
