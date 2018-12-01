import numpy as np
import torch

from ddpg_agent import DDPGAgent
from utils import soft_update, transpose_to_tensor, transpose_list


class MADDPG:
    def __init__(self, action_size, state_size, device, discount_factor=0.95, tau=0.02, num_agents=2):
        super(MADDPG, self).__init__()
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.device = device
        self.num_agents = num_agents
        self.action_size = action_size
        self.state_size = state_size

        self.maddpg_agents = []
        for _ in range(self.num_agents):
            # critic input = state_size + action_size
            # TODO: If not performing well, try adding the actions to the first hidden layer
            self.maddpg_agents.append(
                DDPGAgent(in_actor=state_size, hidden_in_actor=48, hidden_out_actor=24, out_actor=action_size, 
                          in_critic=state_size + action_size, hidden_in_critic=48, hidden_out_critic=24,
                          device=self.device)
            )

        self.iter = 0

    def get_actors(self):
        """Get actors of all the agents in the MADDPG object"""
        return [ddpg_agent.actor for ddpg_agent in self.maddpg_agents]

    def get_target_actors(self):
        """Get target_actors of all the agents in the MADDPG object"""
        return [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agents]

    def act(self, states, noise=0.0):
        """Get actions from all agents in the MADDPG object"""
        states = torch.from_numpy(states).float().to(self.device)
        return [agent.act(state, noise) for agent, state in zip(self.maddpg_agents, states)]

    def target_act(self, states, agent, noise=0.0):
        """Get target network actions from all the agents in the MADDPG object """
        num_states = states.shape[0]
        actions = np.zeros((num_states, self.action_size))
        for i, state in enumerate(states):
            a = agent.target_act(state, noise).cpu().data.numpy()
            actions[i, :] = a
        return actions

    def update(self, experiences, agent_number, logger=None):        
        """Update the critics and actors of all the agents"""
        states, actions, rewards, next_states, dones = experiences
        num_states = states.shape[0]

        agent = self.maddpg_agents[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states, agent)
        target_actions = torch.from_numpy(target_actions).float().to(self.device)
        target_actions = np.clip(target_actions, -1, 1)

        # Adding actions to critic's input
        target_critic_input = torch.cat((next_states, target_actions), dim=1).to(self.device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))

        # actions = torch.cat(actions, dim=1)
        critic_input = torch.cat((states, actions), dim=1).to(self.device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        q_input = np.zeros((num_states, self.action_size))
        
        for i, state in enumerate(states):
            a = agent.actor(state).cpu().data.numpy()
            q_input[i, :] = a
        
        q_input = torch.from_numpy(q_input).float().to(self.device)
        # Combine states and agent's action into input for the critic
        q_input2 = torch.cat((states, q_input), dim=1)
        
        # Get the policy gradient
        actor_loss = - agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

        # if logger:
        #     logger.add_scalars(
        #         'agent%i/losses' % agent_number,
        #         {'critic loss': cl, 'actor_loss': al},
        #         self.iter
        #     )

    def update_targets(self):
        """Soft update the targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
