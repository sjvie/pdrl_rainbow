import math
import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, conv_channels, action_space, num_atoms, device, distributed=True, noisy=True):
        """
        :param input_dim (int): the length of the input vector
        :param action_space (int): the amount of actions
        :param num_atoms (int): the amount of atoms for the probability distribution of each action
        """
        super().__init__()
        self.action_space = action_space
        self.num_atoms = num_atoms
        self.distributed = distributed
        self.noisy = noisy

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=32, kernel_size=8, stride=4, device=device), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, device=device), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, device=device), nn.ReLU()
        )

        sigma_zero = 0.5

        # the size of the output of the convolutional layer
        # 64 * 7 * 7 TODO: checken ob das richtig ist
        self.conv_output_size = 3136
        if self.distributed:
            if self.noisy:
                self.value = nn.Sequential(
                    NoisyLinear(input_dim=self.conv_output_size, output_dim=512, sigma_zero=sigma_zero, device=device),
                    nn.ReLU(),
                    NoisyLinear(input_dim=512, output_dim=num_atoms, sigma_zero=sigma_zero, device=device),
                    # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
                    # nn.Linear(512, 1)
                )

                self.advantage = nn.Sequential(
                    NoisyLinear(input_dim=self.conv_output_size, output_dim=512, sigma_zero=sigma_zero, device=device),
                    nn.ReLU(),
                    NoisyLinear(input_dim=512, output_dim=action_space * num_atoms, sigma_zero=sigma_zero, device=device),
                    # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
                    # nn.Linear(512, action_space)
                )
            else:
                self.value = nn.Sequential(
                    nn.Linear(self.conv_output_size, 512, sigma_zero, device=device),
                    nn.ReLU(),
                    nn.Linear(512, num_atoms, device=device)
                )
                self.advantage = nn.Sequential(
                    nn.Linear(self.conv_output_size, 512, device=device),
                    nn.ReLU(),
                    nn.Linear(512, action_space * num_atoms, device=device)
                )
        else:
            if self.noisy:
                self.value = nn.Sequential(
                    NoisyLinear(input_dim=self.conv_output_size, output_dim=512, sigma_zero=sigma_zero, device=device),
                    nn.ReLU(),
                    NoisyLinear(input_dim=512, output_dim=1, sigma_zero=sigma_zero, device=device),
                    # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
                    # nn.Linear(512, 1)
                )

                self.advantage = nn.Sequential(
                    NoisyLinear(input_dim=self.conv_output_size, output_dim=512, sigma_zero=sigma_zero, device=device),
                    nn.ReLU(),
                    NoisyLinear(input_dim=512, output_dim=action_space, sigma_zero=sigma_zero, device=device),
                    # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
                    # nn.Linear(512, action_space)
                )
            else:
                self.value = nn.Sequential(
                    nn.Linear(self.conv_output_size, 512, device=device),
                    nn.ReLU(),
                    nn.Linear(512, 1, device=device)
                )
                self.advantage = nn.Sequential(
                    nn.Linear(self.conv_output_size, 512, device=device),
                    nn.ReLU(),
                    nn.Linear(512, action_space, device=device)
                )






    def forward(self, x, log=False):
        """
        :param x (Tensor): input of the model. Tensor of dim [input_dim]
        :param log (boolean): whether to calculate the softmax with or without log
        :return (Tensor): output of the model. Tensor of dim [action_space, num_atoms]
        """
        # convolutional layers
        c = self.conv(x)
        c = c.view(-1, self.conv_output_size)

        # value stream (linear layers)
        value = self.value(c)

        # value stream (linear layers)
        advantage = self.advantage(c)
        #duelling + distributed case
        if self.distributed:
            # convert one dimensional tensor to two dimensions
            advantage = advantage.view(-1, self.action_space, self.num_atoms)

            # combine value and advantage stream
            Q_dist = value.unsqueeze(dim=-2) + advantage - advantage.mean(dim=-1, keepdim=True)
            Q_dist = Q_dist.squeeze()

            # apply softmax (with or without log)
            if log:
                return self.log_softmax(Q_dist)
            else:
                return self.softmax(Q_dist)
        #duelling case without distributed
        else:
            q_values = value + (advantage-advantage.mean())
            return q_values


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, device, sigma_zero=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.sigma_zero = sigma_zero

        self.lin_weights = nn.Parameter(torch.empty((output_dim, input_dim), dtype=torch.float32, device=device))
        self.noisy_weights = nn.Parameter(torch.empty((output_dim, input_dim), dtype=torch.float32, device=device))

        self.lin_bias = nn.Parameter(torch.empty(output_dim, dtype=torch.float32, device=device))
        self.noisy_bias = nn.Parameter(torch.empty(output_dim, dtype=torch.float32, device=device))

        # initialize the weights and bias according to section 3.2 in the noisy net paper
        # init linear weights and bias from an independent uniform distribution U[-1/sqrt(p), 1/sqrt(p)]
        lin_init_dist_bounds = math.sqrt(1 / input_dim)
        nn.init.uniform_(self.lin_weights, -lin_init_dist_bounds, lin_init_dist_bounds)
        nn.init.uniform_(self.lin_bias, -lin_init_dist_bounds, lin_init_dist_bounds)

        # init noisy weights and bias to a constant sigma_zero/sqrt(p)
        noisy_init_constant = self.sigma_zero / math.sqrt(input_dim)
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
