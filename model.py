import math
import torch.nn as nn
import torch
from torch.nn import functional as F


class RainbowConv(nn.Module):

    def __init__(self, conf, in_channels):
        super().__init__()
        device = conf.device
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, device=device),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ImpalaConvResBlock(nn.Module):
    def __init__(self, conf, in_channels, hidden_channels, out_channels):
        super().__init__()

        device = conf.device

        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                          device=device))
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                          device=device))
        )

    def forward(self, x):
        r = self.conv1(x)
        r = self.conv2(r)
        return r + x


class ImpalaConvBlock(nn.Module):
    def __init__(self, conf, in_channels, hidden_channels, out_channels):
        super().__init__()

        device = conf.device

        self.block = nn.Sequential(
            # TODO figure out if the padding is correct
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                      padding=1, device=device),
            # TODO figure out if the padding is correct
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ImpalaConvResBlock(conf, in_channels=hidden_channels, hidden_channels=hidden_channels,
                               out_channels=hidden_channels),
            ImpalaConvResBlock(conf, in_channels=hidden_channels, hidden_channels=hidden_channels,
                               out_channels=out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ImpalaConv(nn.Module):
    def __init__(self, conf, in_channels, scale_factor):
        super().__init__()

        self.conv = nn.Sequential(
            ImpalaConvBlock(conf, in_channels, 16 * scale_factor, 16 * scale_factor),
            ImpalaConvBlock(conf, 16 * scale_factor, 32 * scale_factor, 32 * scale_factor),
            ImpalaConvBlock(conf, 32 * scale_factor, 32 * scale_factor, 32 * scale_factor),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class RainbowBody(nn.Module):
    def __init__(self, conf, action_space, in_features, linear_layer):
        super().__init__()

        self.action_space = action_space
        self.num_atoms = conf.distributional_atoms
        self.use_distributional = conf.use_distributional
        self.use_dueling = conf.use_dueling

        device = conf.device

        if not self.use_distributional:
            self.num_atoms = 1

        if self.use_dueling:
            self.value = nn.Sequential(
                linear_layer(in_features=in_features, out_features=512, device=device),
                nn.ReLU(),
                linear_layer(in_features=512, out_features=self.num_atoms, device=device),
            )

            self.advantage = nn.Sequential(
                linear_layer(in_features=in_features, out_features=512, device=device),
                nn.ReLU(),
                linear_layer(in_features=512, out_features=self.action_space * self.num_atoms, device=device),
            )
        else:
            self.final_layers = nn.Sequential(
                linear_layer(in_features=in_features, out_features=512, device=device),
                nn.ReLU(),
                linear_layer(in_features=512, out_features=self.action_space * self.num_atoms, device=device)
            )

    def forward(self, x, log=False):

        if self.use_dueling:
            # value stream (linear layers)
            value = self.value(x)

            # advantage stream (linear layers)
            advantage = self.advantage(x)

            # convert one dimensional tensor to two dimensions
            advantage = advantage.view(-1, self.action_space, self.num_atoms)

            value = value.view(-1, 1, self.num_atoms)

            # combine value and advantage stream
            q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_dist = self.final_layers(x)
            q_dist = q_dist.view(-1, self.action_space, self.num_atoms)

        if self.use_distributional:
            # apply softmax
            if log:
                return F.log_softmax(q_dist, dim=-1)
            else:
                return F.softmax(q_dist, dim=-1)
        else:
            return q_dist.squeeze()


class D2RLBody(nn.Module):
    def __init__(self, conf, action_space, in_features, linear_layer):
        super().__init__()

        assert conf.use_dueling

        self.action_space = action_space
        self.num_atoms = conf.distributional_atoms
        self.use_distributional = conf.use_distributional

        device = conf.device

        if not self.use_distributional:
            self.num_atoms = 1

        hidden_features = 256
        hidden_in_features = hidden_features + in_features

        self.relu = nn.ReLU()

        self.value1 = linear_layer(in_features=in_features, out_features=hidden_features, device=device)
        self.value2 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.value3 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.value4 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.value_out = linear_layer(in_features=hidden_features, out_features=self.num_atoms, device=device)

        self.advantage1 = linear_layer(in_features=in_features, out_features=hidden_features, device=device)
        self.advantage2 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.advantage3 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.advantage4 = linear_layer(in_features=hidden_in_features, out_features=hidden_features, device=device)
        self.advantage_out = linear_layer(in_features=hidden_features, out_features=self.action_space * self.num_atoms,
                                          device=device)

    def _value(self, x):
        v = self.relu(self.value1(x))
        v = torch.cat([x, v], dim=-1)
        v = self.relu(self.value2(v))
        v = torch.cat([x, v], dim=-1)
        v = self.relu(self.value3(v))
        v = torch.cat([x, v], dim=-1)
        v = self.relu(self.value4(v))
        v = self.value_out(v)
        return v

    def _advantage(self, x):
        a = self.relu(self.advantage1(x))
        a = torch.cat([x, a], dim=-1)
        a = self.relu(self.advantage2(a))
        a = torch.cat([x, a], dim=-1)
        a = self.relu(self.advantage3(a))
        a = torch.cat([x, a], dim=-1)
        a = self.relu(self.advantage4(a))
        a = self.advantage_out(a)
        return a

    def forward(self, x, log=False):
        # value stream (linear layers)
        value = self._value(x)

        # advantage stream (linear layers)
        advantage = self._advantage(x)

        # convert one dimensional tensor to two dimensions
        advantage = advantage.view(-1, self.action_space, self.num_atoms)

        value = value.view(-1, 1, self.num_atoms)

        # combine value and advantage stream
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)

        if self.use_distributional:
            # apply softmax
            if log:
                return F.log_softmax(q_dist, dim=-1)
            else:
                return F.softmax(q_dist, dim=-1)
        else:
            return q_dist.squeeze()


class Model(nn.Module):
    def __init__(self, conf, action_space, linear_layer, in_channels):
        super().__init__()

        self.pre, self.body_in_features = self._create_pre(conf, action_space, linear_layer, in_channels)
        self.body = self._create_body(conf, action_space, linear_layer, self.body_in_features)

        self.generate_noise()

    def _create_pre(self, conf, action_space, linear_layer, in_channels):
        raise NotImplementedError

    def _create_body(self, conf, action_space, linear_layer, body_in_features):
        raise NotImplementedError

    def generate_noise(self):
        for noisy_layer in [m for m in self.modules() if isinstance(m, NoisyLinear)]:
            noisy_layer.generate_noise()

    def forward(self, x, log=False):
        c = self.pre(x)
        c = c.view(-1, self.body_in_features)
        return self.body(c, log)


class RainbowModel(Model):
    def _create_pre(self, conf, action_space, linear_layer, in_channels):
        conv = RainbowConv(conf, in_channels)
        # the size of the output of the convolutional layer (given frame_width = frame_height = 84)
        # 64 * 7 * 7
        conv_out_features = 3136
        return conv, conv_out_features

    def _create_body(self, conf, action_space, linear_layer, body_in_features):
        return RainbowBody(conf, action_space, body_in_features, linear_layer)


class D2RLModel(Model):
    def _create_pre(self, conf, action_space, linear_layer, in_channels):
        conv = RainbowConv(conf, in_channels)
        # the size of the output of the convolutional layer (given frame_width = frame_height = 84)
        # 64 * 7 * 7
        conv_out_features = 3136
        return conv, conv_out_features

    def _create_body(self, conf, action_space, linear_layer, body_in_features):
        return D2RLBody(conf, action_space, body_in_features, linear_layer)


class D2RLImpalaModel(Model):
    def _create_pre(self, conf, action_space, linear_layer, in_channels):
        scale_factor = conf.model_pre_scale_factor
        adaptive_pool_size = conf.impala_adaptive_pool_size
        conv = nn.Sequential(
            ImpalaConv(conf, in_channels, scale_factor),
            nn.AdaptiveMaxPool2d((adaptive_pool_size, adaptive_pool_size))
        )
        # the size of the output of the convolutional layer (independent of frame size because of adaptive max pooling)
        conv_out_features = 32 * adaptive_pool_size * adaptive_pool_size * scale_factor
        return conv, conv_out_features

    def _create_body(self, conf, action_space, linear_layer, body_in_features):
        return D2RLBody(conf, action_space, body_in_features, linear_layer)


class ImpalaModel(Model):
    def _create_pre(self, conf, action_space, linear_layer, in_channels):
        scale_factor = conf.model_pre_scale_factor
        adaptive_pool_size = conf.impala_adaptive_pool_size
        conv = nn.Sequential(
            ImpalaConv(conf, in_channels, scale_factor),
            nn.AdaptiveMaxPool2d((adaptive_pool_size, adaptive_pool_size))
        )
        # the size of the output of the convolutional layer (independent of frame size because of adaptive max pooling)
        conv_out_features = 32 * adaptive_pool_size * adaptive_pool_size * scale_factor
        return conv, conv_out_features

    def _create_body(self, conf, action_space, linear_layer, body_in_features):
        return RainbowBody(conf, action_space, body_in_features, linear_layer)


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

        self.e_weights = torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        self.e_bias = torch.empty(out_features, dtype=torch.float32, device=device)

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

        return F.linear(x, self.lin_weights + self.noisy_weights * self.e_weights,
                        self.lin_bias + self.noisy_bias * self.e_bias)

    def generate_noise(self):
        self.e_weights, self.e_bias = self.get_eps_weight_bias()

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
        e_i = torch.randn(self.input_dim, device=self.device)
        e_i = self.eps_function(e_i)

        # get second random vector and apply function
        e_j = torch.randn(self.output_dim, device=self.device)
        e_j = self.eps_function(e_j)

        # combine vectors to get the epsilon matrix
        e_mat = torch.outer(e_j, e_i)

        # return the matrix and the second vector
        return e_mat, e_j
