import functools
import math
import torch.nn as nn
import torch
from torch.nn import functional as F


class Model(nn.Module):

    def __init__(self, action_space, device, conf, conv_channels=None, input_features=None):
        super().__init__()

        self.action_space = action_space
        self.num_atoms = conf.distributional_atoms
        self.use_distributional = conf.use_distributional
        self.use_noisy = conf.use_noisy
        self.use_dueling = conf.use_dueling

        self.use_conv = conf.use_conv

        if conf.use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=conv_channels, out_channels=32, kernel_size=8, stride=4, device=device),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, device=device),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, device=device),
                nn.ReLU()
            )
            # the size of the output of the convolutional layer
            # 64 * 7 * 7
            self.conv_output_size = 3136
        else:
            self.conv = nn.Linear(in_features=input_features, out_features=128, device=device)
            self.conv_output_size = 128

        sigma_zero = conf.noisy_sigma_zero

        if self.use_noisy:
            layer = functools.partial(NoisyLinear, sigma_zero=sigma_zero)
        else:
            layer = nn.Linear

        if not self.use_distributional:
            self.num_atoms = 1

        if self.use_dueling:
            self.value = nn.Sequential(
                layer(in_features=self.conv_output_size, out_features=512, device=device),
                nn.ReLU(),
                layer(in_features=512, out_features=self.num_atoms, device=device),
            )

            self.advantage = nn.Sequential(
                layer(in_features=self.conv_output_size, out_features=512, device=device),
                nn.ReLU(),
                layer(in_features=512, out_features=action_space * self.num_atoms, device=device),
            )
        else:
            self.final_layers = nn.Sequential(
                layer(in_features=self.conv_output_size, out_features=512, device=device),
                nn.ReLU(),
                layer(in_features=512, out_features=action_space * self.num_atoms, device=device)
            )

    def forward(self, x, log):

        # convolutional layers
        c = self.conv(x)
        c = c.view(-1, self.conv_output_size)

        if self.use_dueling:
            # value stream (linear layers)
            value = self.value(c)

            # advantage stream (linear layers)
            advantage = self.advantage(c)

            # convert one dimensional tensor to two dimensions
            advantage = advantage.view(-1, self.action_space, self.num_atoms)

            value = value.view(-1, 1, self.num_atoms)

            # combine value and advantage stream
            q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_dist = self.final_layers(c)
            q_dist = q_dist.view(-1, self.action_space, self.num_atoms)

        if self.use_distributional:
            # apply softmax
            if log:
                return F.log_softmax(q_dist, dim=-1)
            else:
                return F.softmax(q_dist, dim=-1)
        else:
            return q_dist


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, device, sigma_zero=0.5):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.device = device
        self.sigma_zero = sigma_zero

        self.lin_weights = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32, device=device))
        self.noisy_weights = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32, device=device))

        self.lin_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))
        self.noisy_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))

        # initialize the weights and bias according to section 3.2 in the noisy net paper
        # init linear weights and bias from an independent uniform distribution U[-1/sqrt(p), 1/sqrt(p)]
        lin_init_dist_bounds = math.sqrt(1 / in_features)
        nn.init.uniform_(self.lin_weights, -lin_init_dist_bounds, lin_init_dist_bounds)
        nn.init.uniform_(self.lin_bias, -lin_init_dist_bounds, lin_init_dist_bounds)

        # init noisy weights and bias to a constant sigma_zero/sqrt(p)
        noisy_init_constant = self.sigma_zero / math.sqrt(in_features)
        nn.init.constant_(self.noisy_weights, noisy_init_constant)
        nn.init.constant_(self.noisy_bias, noisy_init_constant)

    def forward(self, x):
        """
        :param x (Tensor): input of the layer. Tensor of dim [batch_size, input_dim] or [input_dim]
        :return (Tensor): output of the layer. Tensor of dim [output_dim]
        """

        # calculate the linear part of the layer using the linear weights and bias
        lin = torch.matmul(self.lin_weights, x.transpose(0, -1)).transpose(0, -1) + self.lin_bias

        # get the random noise values
        e_weights, e_bias = self.get_eps_weight_bias()

        # calculate the noisy part of the layer
        noisy_bias_e = self.noisy_bias * e_bias
        noisy_weights_e = self.noisy_weights * e_weights
        noisy = torch.matmul(noisy_weights_e, x.transpose(0, -1)).transpose(0, -1) + noisy_bias_e

        # combine linear and noisy part
        return lin + noisy

    # f(x) = sgn(x)* Sqrt(|x|)  from noisy net paper (page 5 under eq. 11)
    def eps_function(self, x):
        return x.sign() * x.abs().sqrt()

    def get_eps_weight_bias(self):
        """
        gets the random epsilon values (with applied f(x)) using the factorised Gaussian noise approach

        :return (Tensor, Tensor): the random epsilon for the weights and the bias. Tensors of dim
                                  [output_dim, input_dim] and [output_dim]
        """

        # get first random vector and apply function
        e_i = torch.empty(self.input_dim, device=self.device)
        e_i.normal_()
        e_i = self.eps_function(e_i)

        # get second random vector and apply function
        e_j = torch.empty(self.output_dim, device=self.device)
        e_j.normal_()
        e_j = self.eps_function(e_j)

        # combine vectors to get the epsilon matrix
        e_mat = torch.outer(e_j, e_i)

        # return the matrix and the second vector
        return e_mat, e_j
