import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Net(torch.nn.Module):

    def __init__(self, num_channels, num_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, 8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=2)

        self.lstm = nn.LSTMCell(4608, 256)

        num_outputs = num_actions
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, req_params=None):
        # if requested params is list they will be returned as well 
        inputs, (hx, cx) = inputs
        conv1_out = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(conv1_out))
        x = x.view(x.size()[0], -1)  # first dimension is batch size
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        # quick hardcode req_params
        if req_params is None or 'conv1_out' not in req_params:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        else:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx, conv1_out)

