import copy
import os
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from MADDPG.hyperparameters import *
from MADDPG.model import Actor, Critic
from MADDPG.ou_noise import OUNoise
from MADDPG.replay_buffer import ReplayBuffer


class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, agent_id):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in the environment
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.agent_id = agent_id

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size)
        
        self.noise_amplification = NOISE_AMPLIFICATION
        self.noise_amplification_decay = NOISE_AMPLIFICATION_DECAY
        
        self._print_network()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            self._decay_noise_amplification()
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, agent_number, next_actions, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        agent_id_tensor = torch.tensor([self.agent_id - 1]).to(device)

        ### Update critic
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)        
        Q_targets = rewards.index_select(1, agent_id_tensor) + (GAMMA * Q_targets_next * (1 - dones.index_select(1, agent_id_tensor)))
        
        Q_expected = self.critic_local(states, actions)
        
        # Minimize the loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


        ### Update actor
        self.actor_optimizer.zero_grad()
        
        # Minimize the loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()


        ### Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def _print_network(self):
        print("Agent #{}".format(self.agent_id))
        print("Actor (Local):")
        print(self.actor_local)
        print("Actor (Target):")
        print(self.actor_target)
        print("Critic (Local):")
        print(self.critic_local)
        print("Critic (Target):")
        print(self.critic_target)
        if self.agent_id != NUM_AGENTS:
            print("_______________________________________________________________")
            
    def _decay_noise_amplification(self):
        self.noise_amplification *= self.noise_amplification_decay
        #if self.noise_amplification < 1:
        #    self.noise_amplification = 1 