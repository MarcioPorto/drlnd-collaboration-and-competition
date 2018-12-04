import numpy as np
import torch
import torch.nn.functional as F

from ddpg_agent import DDPGAgent
from utils import soft_update, transpose_to_tensor, transpose_list


class MADDPG:
    def __init__(self, action_size, observation_size, device, lr_actor, lr_critic,
                 discount_factor=0.95, tau=0.02, num_agents=2):
        super(MADDPG, self).__init__()
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.device = device
        self.num_agents = num_agents
        self.action_size = action_size
        self.observation_size = observation_size

        # Critic needs observation from all agents and all individual actions (added in the second hidden layer)
        in_critic = (observation_size + action_size) * self.num_agents
        # in_critic = observation_size * self.num_agents
        critic_action_size = action_size * self.num_agents

        self.maddpg_agents = []
        for _ in range(self.num_agents):
            self.maddpg_agents.append(
                DDPGAgent(in_actor=observation_size, out_actor=action_size,
                          in_critic=in_critic, critic_action_size=critic_action_size,
                          device=self.device, lr_actor=lr_actor, lr_critic=lr_critic)
            )

        self.iter = 0

    def get_actors(self):
        """Get actors of all the agents in the MADDPG object"""
        return [ddpg_agent.actor for ddpg_agent in self.maddpg_agents]

    def get_target_actors(self):
        """Get target_actors of all the agents in the MADDPG object"""
        return [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agents]

    def take_action(self, observations, noise=False):
        """Get actions from all agents in the MADDPG object"""
        observations = torch.from_numpy(observations).float().to(self.device)        
        return [agent.take_action(observation, noise) for agent, observation in zip(self.maddpg_agents, observations)]

    def act(self, observations_full, noise=False):
        """Get actions from a given agent in the MADDPG object """
        num_observations = observations_full.shape[0]
        actions = np.zeros((num_observations, self.num_agents * self.action_size))
        observations_full = observations_full.reshape(-1, 2, 24)

        for a_i, agent in enumerate(self.maddpg_agents):
            for i, observation in enumerate(observations_full[a_i]):
                start = a_i * 2
                end = start + 2
                actions[i, start:end] = agent.act(observation, noise).cpu().data.numpy()
        
        return torch.from_numpy(actions).float().to(self.device)

    def target_act(self, observations_full, noise=False):
        """Get target network actions from a given agent in the MADDPG object """
        num_observations = observations_full.shape[0]
        actions = np.zeros((num_observations, self.num_agents * self.action_size))
        observations_full = observations_full.reshape(-1, 2, 24)

        for a_i, agent in enumerate(self.maddpg_agents):
            for i, observation in enumerate(observations_full[a_i]):
                start = a_i * 2
                end = start + 2
                actions[i, start:end] = agent.target_act(observation, noise).cpu().data.numpy()

        return torch.from_numpy(actions).float().to(self.device)

    def update(self, experiences, agent_number, logger=None):        
        """Update the critics and actors of all the agents"""
        (observations, observations_full, actions, actions_full, 
         rewards, next_observations, next_observations_full, dones) = experiences

        num_observations = observations.shape[0]
        agent = self.maddpg_agents[agent_number]

        ### UPDATE CRITIC
        agent.critic_optimizer.zero_grad()
        next_actions_full = self.target_act(next_observations_full)
        target_critic_input = torch.cat((next_observations_full, next_actions_full), dim=1).to(self.device)
        with torch.no_grad():
            # q_next = agent.target_critic(next_observations_full, next_actions_full)
            q_next = agent.target_critic(target_critic_input)
        y = rewards + (self.discount_factor * q_next * (1 - dones))
        
        # q = agent.critic(observations_full, actions_full)
        critic_input = torch.cat((observations_full, actions_full), dim=1).to(self.device)
        q = agent.critic(critic_input)

        # Minimize the loss
        critic_loss = F.mse_loss(q, y)
        critic_loss.backward()
        agent.critic_optimizer.step()

        ### UPDATE ACTOR
        agent.actor_optimizer.zero_grad()
        actions_pred_full = self.act(observations_full)
        critic_input = torch.cat((observations_full, actions_pred_full), dim=1)
        
        # Minimize the loss
        # actor_loss = - agent.critic(observations_full, actions_pred_full).mean()
        actor_loss = - agent.critic(critic_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

        ### LOGGING
        if logger:
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()

            logger.add_scalars(
                'agent%i/losses' % agent_number,
                {'critic loss': cl, 'actor_loss': al},
                self.iter
            )

    def update_targets(self):
        """Soft update the targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
