import torch
import numpy as np

from model import Network
from ou_noise import OUNoise
from utils import hard_update, gumbel_softmax, onehot_from_logits


class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, 
                 in_critic, hidden_in_critic, hidden_out_critic, device, 
                 lr_actor=1e-4, lr_critic=1e-3):
        super(DDPGAgent, self).__init__()
        self.device = device

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, out_actor).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, out_actor).to(device)

        self.noise = OUNoise(size=out_actor, seed=1)
        
        # Initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, state, noise=0.0):
        state = state.to(self.device)
        actions = self.actor(state) + noise * self.noise.sample()
        return torch.clamp(actions, -1, 1)

    def target_act(self, state, noise=0.0):
        state = state.to(self.device)
        actions = self.target_actor(state) + noise * self.noise.sample()
        return torch.clamp(actions, -1, 1)
