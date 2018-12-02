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

        self.maddpg_agents = []
        for _ in range(self.num_agents):
            self.maddpg_agents.append(
                DDPGAgent(in_actor=observation_size, hidden_in_actor=600, hidden_out_actor=400, out_actor=action_size, 
                        #   in_critic=observation_size, hidden_in_critic=600, hidden_out_critic=400,
                          in_critic=observation_size + action_size, hidden_in_critic=600, hidden_out_critic=400,
                          device=self.device, lr_actor=lr_actor, lr_critic=lr_critic)
            )

        self.iter = 0

    def get_actors(self):
        """Get actors of all the agents in the MADDPG object"""
        return [ddpg_agent.actor for ddpg_agent in self.maddpg_agents]

    def get_target_actors(self):
        """Get target_actors of all the agents in the MADDPG object"""
        return [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agents]

    def act(self, observations, noise=0.0):
        """Get actions from all agents in the MADDPG object"""
        observations = torch.from_numpy(observations).float().to(self.device)
        return [agent.act(observation, noise) for agent, observation in zip(self.maddpg_agents, observations)]

    def target_act(self, observations, agent, noise=0.0):
        """Get target network actions from all the agents in the MADDPG object """
        num_observations = observations.shape[0]
        actions = np.zeros((num_observations, self.action_size))
        for i, observation in enumerate(observations):
            actions[i, :] = agent.target_act(observation, noise).cpu().data.numpy()
        return actions

    def update(self, experiences, agent_number, logger=None):        
        """Update the critics and actors of all the agents"""
        observations, observations_full, actions, rewards, next_observations, next_observations_full, dones = experiences
        num_observations = observations.shape[0]
        agent = self.maddpg_agents[agent_number]

        ### UPDATE CRITIC
        target_actions = self.target_act(next_observations, agent)
        target_actions = torch.from_numpy(target_actions).float().to(self.device)
        target_critic_input = torch.cat((next_observations, target_actions), dim=1).to(self.device)

        with torch.no_grad():
            # q_next = agent.target_critic(next_observations, target_actions)
            q_next = agent.target_critic(target_critic_input)

        y = rewards + (self.discount_factor * q_next * (1 - dones))
        # q = agent.critic(observations, actions)
        critic_input = torch.cat((observations, actions), dim=1).to(self.device)
        q = agent.critic(critic_input)

        critic_loss = F.mse_loss(q, q_next)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        ### UPDATE ACTOR
        # make input to agent
        # TODO: Refactor this into its own function?
        q_input = np.zeros((num_observations, self.action_size))
        for i, observation in enumerate(observations):
            q_input[i, :] = agent.actor(observation).cpu().data.numpy()
        q_input = torch.from_numpy(q_input).float().to(self.device)

        q_input2 = torch.cat((observations, q_input), dim=1)
        
        # Get the policy gradient
        # actor_loss = - agent.critic(observations, q_input).mean()
        actor_loss = - agent.critic(q_input2).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
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
