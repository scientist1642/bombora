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

    def __init__(self, input_shape, num_actions):
        super(Net, self).__init__()
        in_number = input_shape[1]
        self.fc1 = nn.Linear(in_number, 32)
        self.fc2 = nn.Linear(32, 16)
        self.critic_linear = nn.Linear(16, 1)
        self.actor_linear = nn.Linear(16, num_actions)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, req_params=None):
        # if requested params is list they will be returned as well 
        #import ipdb; ipdb.set_trace()
        inputs, (hx, cx) = inputs
        x = inputs.view(inputs.size()[0], -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        conv1_out = x
        
        # quick hardcode req_params
        if req_params is None or 'conv1_out' not in req_params:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        else:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx, conv1_out)

