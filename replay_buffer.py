import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        field_names = ["observation", "observation_full", "action", "reward", "next_observation", "next_observation_full", "done"]

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=field_names)
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, observation, observation_full, action, reward, next_observation, next_observation_full, done):
        """Add a new experience to memory."""
        self.memory.append(
            self.experience(observation, observation_full, action, reward, next_observation, next_observation_full, done)
        )
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        observations = torch.from_numpy(np.vstack([e.observation for e in experiences if e is not None])).float().to(self.device)
        observations_full = torch.from_numpy(np.vstack([e.observation_full for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_observations = torch.from_numpy(np.vstack([e.next_observation for e in experiences if e is not None])).float().to(self.device)
        next_observations_full = torch.from_numpy(np.vstack([e.next_observation_full for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (observations, observations_full, actions, rewards, next_observations, next_observations_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
