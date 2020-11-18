import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


DQN_MODELS = {}  # add new DQN_MODELS to the dict after class definition. Ex: DQN_DQN_MODELS['dqn'] = DQN


# Four layers DQN with Dropout
class DQN(nn.Module):
    def __init__(self, input_shape, out_shape, n_hidden, dropout_p=0.01):
        super(DQN, self).__init__()

        self.input_dims = input_shape.shape[0]
        self.hidden_dims = n_hidden
        self.output_dims = out_shape.n

        self.fc1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc4 = nn.Linear(self.hidden_dims, self.output_dims)
        self.rl = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims)
        x = self.dropout(self.rl(self.fc1(x)))
        x = self.rl(self.fc2(x))
        x = self.rl(self.fc3(x))
        return self.fc4(x.view(x.size(0), -1))


DQN_MODELS['dqn'] = DQN


# Four layers Dueling DQN with Dropout
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, out_shape, n_hidden, dropout_p=0.01):
        super(DuelingDQN, self).__init__()

        self.input_dims = input_shape.shape[0]
        self.hidden_dims = n_hidden
        self.output_dims = out_shape.n

        self.fcIn = nn.Linear(self.input_dims, self.hidden_dims)
        self.rl = nn.ReLU()
        self.fcH1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fcH2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fcVal = nn.Linear(self.hidden_dims, 1)
        self.fcAdv = nn.Linear(self.hidden_dims, self.output_dims)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims)
        x = self.rl(self.fcIn(x))
        x = self.dropout(self.rl(self.fcH1(x)))
        x = self.rl(self.fcH2(x))
        hidden_out = x.view(x.size(0), -1)
        val = self.fcVal(hidden_out)
        adv = self.fcAdv(hidden_out)
        return val + adv - adv.mean()


DQN_MODELS['dueling_dqn'] = DuelingDQN
# 'noisy_dqn': NoisyDQN,
# 'conv_dqn': ConvDQN


# Four layers DQN with two noisy layers (the two last layers):
class NoisyDQN(nn.Module):
    def __init__(self, input_shape, out_shape, n_hidden):
        super(NoisyDQN, self).__init__()

        self.input_dims = input_shape.shape[0]
        self.hidden_dims = n_hidden
        self.output_dims = out_shape.n

        self.fc_linear = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU()
        )

        self.noisy_layers = [
            NoisyFactorizedLinear(self.hidden_dims, self.hidden_dims),
            NoisyFactorizedLinear(self.hidden_dims, self.output_dims)
        ]

        self.fc_noisy_linear = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims)
        x = self.fc_linear(x)
        x = self.fc_noisy_linear(x)
        return x

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


DQN_MODELS['noisy_dqn'] = NoisyDQN


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class ConvDQN(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(ConvDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_shape)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


DQN_MODELS['conv_dqn'] = ConvDQN
