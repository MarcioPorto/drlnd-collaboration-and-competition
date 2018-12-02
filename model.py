import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, 
                 actor=False, activation_fn=f.relu):
        """Network definition shared by actor and critic"""
        super(Network, self).__init__()

        self.activation_fn = activation_fn
        self.actor = actor
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)

        # if self.actor:
        #     self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        # else:
        #     # add action to second layer in the critic
        #     self.fc2 = nn.Linear(hidden_in_dim + output_dim, hidden_out_dim)

        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)

        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state, action=None):
        h1 = self.activation_fn(self.fc1(state))
        h2 = self.activation_fn(self.fc2(h1))
        h3 = self.fc3(h2)
        # if self.actor:
        #     h1 = self.activation_fn(self.fc1(state))
        #     h2 = self.activation_fn(self.fc2(h1))
        #     h3 = self.fc3(h2)
        # else:
        #     h1 = self.activation_fn(self.fc1(state))
        #     h1a = torch.cat((h1, action), dim=1)
        #     h2 = self.activation_fn(self.fc2(h1a))
        #     h3 = self.fc3(h2)
        return h3
