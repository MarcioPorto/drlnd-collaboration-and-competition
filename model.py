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
                 actor=False):
        """Network definition shared by actor and critic"""
        super(Network, self).__init__()
        self.actor = actor
        
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        h1 = f.relu(self.fc1(state))
        h2 = f.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        if self.actor:
            h3 = torch.tanh(h3)
        return h3


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_in_dim=600, hidden_out_dim=400):
        """Actor model definition"""
        super(Actor, self).__init__()
        self.actor = True
        
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        h1 = f.relu(self.fc1(state))
        h2 = f.relu(self.fc2(h1))
        h3 = torch.tanh(h2)
        return h3

class Critic(nn.Module):
    def __init__(self, input_dim, critic_action_size, hidden_in_dim=600, hidden_out_dim=400):
        """Critic model definition"""
        super(Critic, self).__init__()
        self.actor = False
        
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim + critic_action_size, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state, action):
        h1 = self.relu(self.fc1(state))
        h1a = torch.cat((h1, action), dim=1)
        h2 = self.relu(self.fc2(h1a))
        h3 = self.fc3(h2)
        return h3
