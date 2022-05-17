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

        sigma_zero = 0.5

        self.value = nn.Sequential(
            NoisyLinear(input_dim=64, output_dim=512, sigma_zero=sigma_zero), nn.ReLU(),
            NoisyLinear(input_dim=512, output_dim=1, sigma_zero=sigma_zero),
            # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
            # nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(input_dim=64, output_dim=512, sigma_zero=sigma_zero), nn.ReLU(),
            NoisyLinear(input_dim=512, output_dim=action_space, sigma_zero=sigma_zero),
            # nn.Linear(in_channels=64, out_channels=512), nn.ReLU(),
            # nn.Linear(512, action_space)
        )

    def forward(self, input):
        c = self.conv(input)
        value = self.value(c)
        advantage = self.advantage(c)
        Q = value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return Q

    def select_action(self, state):
        self.eval()
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

    # TODO


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, sigma_zero=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_zero = sigma_zero

        self.v_eps_function = np.vectorize(self.eps_function)

        self.lin_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.noisy_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.lin_bias = nn.Parameter(torch.Tensor(output_dim))
        self.noisy_bias = nn.Parameter(torch.Tensor(output_dim))

        lin_init_dist_bounds = math.sqrt(3 / input_dim)
        nn.init.uniform_(self.lin_weights, -lin_init_dist_bounds, lin_init_dist_bounds)
        nn.init.uniform_(self.lin_bias, -lin_init_dist_bounds, lin_init_dist_bounds)

        noisy_init_constant = self.sigma_zero / math.sqrt(input_dim)
        nn.init.constant_(self.noisy_weights, noisy_init_constant)
        nn.init.constant_(self.noisy_bias, noisy_init_constant)

    def forward(self, x):
        lin = torch.add(torch.mm(self.lin_weights, x), self.lin_bias)

        e_weights, e_bias = self.get_eps_weight_bias(x)
        noisy_bias_e = torch.mul(self.noisy_bias, e_bias)
        noisy_weights_e = torch.mul(self.noisy_weights, e_weights)
        noisy = torch.add(torch.mm(noisy_weights_e, x), noisy_bias_e)

        return torch.add(lin, noisy)

    # f(x) = sgn(x)* Sqrt(|x|)  from noisy net paper (page 5 under eq. 11)
    def eps_function(self, x):
        return np.sign(x) * np.sqrt(np.abs(x))

    # epsilon_j könnte ev auch unterschiedlich sein, unklar aus paper (noisy net  eq 10 & 11)
    def get_eps_weight_bias(self, x):
        lfunc = lambda e: self.eps_function(e)
        e_i_np = np.random.randn(self.input_dim)
        e_i_np = lfunc(e_i_np)

        e_j_np = np.random.randn(self.output_dim)  # q dim
        e_j_np = lfunc(self.eps_function(e_j_np))
        e_mat_np = e_i_np.matmul(e_j_np)

        e_mat = torch.as_tensor(e_mat_np, device='cuda')
        e_bias = torch.as_tensor(e_j_np, device='cuda')
        return e_mat, e_bias
