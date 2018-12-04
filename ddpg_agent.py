import torch
import numpy as np

from model import Network, Actor, Critic
from ou_noise import OUNoise
from utils import hard_update, gumbel_softmax, onehot_from_logits


class DDPGAgent:
    def __init__(self, in_actor, out_actor, in_critic, critic_action_size, 
                 device, hidden_in_actor=400, hidden_out_actor=300, hidden_in_critic=400, 
                 hidden_out_critic=300, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0.0):
        super(DDPGAgent, self).__init__()
        self.device = device
        self.action_size = out_actor

        # self.actor = Actor(in_actor, out_actor, hidden_in_actor, hidden_out_actor).to(device)
        # self.critic = Critic(in_critic, critic_action_size, hidden_in_critic, hidden_out_critic).to(device)
        # self.target_actor = Actor(in_actor, out_actor, hidden_in_actor, hidden_out_actor).to(device)
        # self.target_critic = Critic(in_critic, critic_action_size, hidden_in_critic, hidden_out_critic).to(device)

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(size=out_actor, seed=1)
        
        # Initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    def take_action(self, state, noise=False):
        state = state.to(self.device)
        actions = np.zeros(self.action_size)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy() 
        self.actor.train()
        if noise:
            actions += self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return torch.from_numpy(actions).float().to(self.device)

    def act(self, state, noise=False):
        state = state.to(self.device)
        actions = self.actor(state)
        if noise:
            actions += self.noise.sample()
        return torch.clamp(actions, -1, 1)

    def target_act(self, state, noise=False):
        state = state.to(self.device)
        actions = self.target_actor(state)
        if noise:
            actions += self.noise.sample()
        return torch.clamp(actions, -1, 1)
