import os

import imageio  # for saving gif
import matplotlib.pyplot as plt
import progressbar as pb
import torch
import numpy as np
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from replay_buffer import ReplayBuffer
from maddpg_agent import MADDPG
from utils import transpose_list, transpose_to_tensor


### ADVICE ###
# TODO: If not solved fast enough, play around with tau and batch_size in order to finish faster
# TODO: Maybe try to add in states and actions in the second layer (like DDPG)
# TODO: If input values are far from 1, use batch norm; but if they're only a bit bigger it shouldn't matter too much

### QUESTIONS ###
# TODO: Maybe parallel agents is 4 and normal agents are 3 because one of them is the adversary


class TennisPlayingModel:
    def __init__(self, environment, num_agents=2, num_episodes=1000, save_gifs=False):
        seed = 1
        self.seeding(seed=seed)

        # Environment
        self.env = environment
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.num_agents = len(env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.state_size = env_info.vector_observations.shape[1]

        self.agent_rewards = [[] for i in range(self.num_agents)]
        
        # Hyperparameters
        self.number_of_episodes = num_episodes
        self.episode_length = 80                        # max_t equivalent
        self.batch_size = 1000
        self.save_interval = 1000                       # how many episodes to save policy and gif
        self.episode_per_update = 2                     # how many episodes before update
        self.episodes_in_replay = 5000

        # Helpers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maddpg = MADDPG(action_size=self.action_size, state_size=self.state_size, device=self.device)
        self.buffer = ReplayBuffer(
            action_size=self.action_size, 
            buffer_size=int(self.episodes_in_replay * self.episode_length), 
            batch_size=self.batch_size, 
            seed=seed,
            device=self.device
        )

        # Exploration noise
        # Controls how much OU Noise is applied
        self.noise = 2
        self.noise_reduction = 0.9999

        # Options
        self.save_gifs = save_gifs

        # Model
        self.model_dir = os.getcwd() + "/saved_models"
        os.makedirs(self.model_dir, exist_ok=True)

        # # Logging
        # self.log_dir = os.getcwd() + "/log"
        # self.logger = SummaryWriter(log_dir=self.log_dir)

    def seeding(self, seed=1):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def save_model(self, episode):
        for i in range(self.num_agents):
            torch.save(
                self.maddpg.maddpg_agents[i].actor.state_dict(), 
                os.path.join(self.model_dir, 'actor_params-agent-{}.pth'.format(i))
            )
            torch.save(
                self.maddpg.maddpg_agents[i].actor_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'actor_optim_params-agent-{}.pth'.format(i))
            )
            torch.save(
                self.maddpg.maddpg_agents[i].critic.state_dict(), 
                os.path.join(self.model_dir, 'critic_params-agent-{}.pth'.format(i))
            )
            torch.save(
                self.maddpg.maddpg_agents[i].critic_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'critic_optim_params-agent-{}.pth'.format(i))
            )

        if self.save_gifs:
            imageio.mimsave(
                os.path.join(self.model_dir, 'episode-{}.gif'.format(episode)), frames, duration=.04
            )
    
    def train(self):
        widget = [
            "Episode: ", pb.Counter(), '/' , str(self.number_of_episodes), ' ', 
            pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' '
        ]
        timer = pb.ProgressBar(widgets=widget, maxval=self.number_of_episodes).start()
        self.agent_rewards = [[] for i in range(self.num_agents)]  # Clear agent rewards

        for episode in range(1, self.number_of_episodes+1):
            timer.update(episode)

            # Reset OU Noise
            for i in range(self.num_agents):
                self.maddpg.maddpg_agents[i].noise.reset()

            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            episode_rewards = np.zeros(self.num_agents)  # rewards this episode (all 0's initially)

            # Save info if once every save_interval steps or if last episode
            save_info = ((episode) % self.save_interval == 0 or episode == self.number_of_episodes)

            for episode_t in range(self.episode_length):
                actions = self.maddpg.act(states, noise=self.noise)
                actions_array = torch.stack(actions).detach().numpy()
                actions = np.rollaxis(actions_array, 1)
                actions = np.clip(actions, -1, 1)
                self.noise *= self.noise_reduction

                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                for i in range(self.num_agents):
                    self.buffer.add(states[i,:], actions[i,:], rewards[i], next_states[i,:], dones[i])

                episode_rewards += rewards
                states = next_states

                # Ignoring this for now so the agent keeps playing
                # if np.any(dones):
                #     break
            
            # Update once after every episode_per_update
            if len(self.buffer) > self.batch_size and episode % self.episode_per_update == 0:
                for a_i in range(self.num_agents):
                    samples = self.buffer.sample()
                    # self.maddpg.update(samples, a_i, self.logger)
                    self.maddpg.update(samples, a_i, None)
                self.maddpg.update_targets()
            
            # Save rewards from this episode
            for i in range(self.num_agents):
                self.agent_rewards[i].append(episode_rewards[i])

            if episode % 100 == 0 or episode == self.number_of_episodes:
                avg_rewards = []
                for i in range(self.num_agents):
                    avg_rewards.append(np.mean(self.agent_rewards[i]))
                
                # # Clear rewards???
                # self.agent_rewards = [[] for i in range(self.num_agents)]

                # TODO: Add calculation for 100-episodes rolling reward average
                
                # for a_i, avg_rew in enumerate(avg_rewards):
                #     self.logger.add_scalar("agent%i/mean_episode_rewards" % a_i, avg_rew, episode)

            if save_info:
                self.save_model(episode)
        
        # TODO: Add a way to only close these when desired
        # self.terminate_env()
        timer.finish()

    def terminate_env(self):
        self.env.close()
        # self.logger.close()
    
    def plot_training_progress(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot one line for each agent
        for i in range(self.num_agents):
            scores = self.agent_rewards[i]
            plt.plot(np.arange(1, len(scores) + 1), scores, label="Agent #{}".format(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()

    def test(self, num_games=5, load_model=False):
        if load_model:
            for i in range(self.num_agents):
                self.maddpg.maddpg_agents[i].actor.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'actor_params-agent-{}.pth'.format(i)))
                )
                self.maddpg.maddpg_agents[i].actor_optimizer.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'actor_optim_params-agent-{}.pth'.format(i)))
                )
                self.maddpg.maddpg_agents[i].critic.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'critic_params-agent-{}.pth'.format(i)))
                )
                self.maddpg.maddpg_agents[i].critic_optimizer.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'critic_optim_params-agent-{}.pth'.format(i)))
                )

        for i in range(1, num_games+1):
            env_info = self.env.reset(train_mode=False)[self.brain_name]   
            states = env_info.vector_observations
            scores = np.zeros(self.num_agents)
            
            while True:
                actions = self.maddpg.act(states)
                actions_array = torch.stack(actions).detach().numpy()
                actions = np.rollaxis(actions_array, 1)
                actions = np.clip(actions, -1, 1)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                scores += env_info.rewards
                states = next_states
                
                if np.any(dones):
                    break
            
            print("Score (max over agents) from episode {}: {}".format(i, np.max(scores)))
