import torch


# TODO: Add hyperparameters
RANDOM_SEED = 0
# # # 200 x 150
# BUFFER_SIZE = int(1e6)  # replay buffer size
# BATCH_SIZE = 512        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-2              # for soft update of target parameters
# LR_ACTOR = 1e-3         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0.0      # L2 weight decay
# UPDATE_EVERY = 4
# NUM_AGENTS = num_agents
# STATE_SIZE = int(state_size / 3)  # this environment has 3 time frames stacked together as its state
# ACTION_SIZE = action_size
# NOISE_AMPLIFICATION = 2
# NOISE_AMPLIFICATION_DECAY = 0.9999

# no difference between
# 200x150 > 128x128
# 1e-1 > 5e-2 TAU
# 2, 0.9999 > 1, 1 NOISE AMP
# 512 > 1024 BATCH_SIZE

# 5e-4 LRs works quite well
# 5e-4, 2e-3

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 5e-2              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_EVERY = 2
NUM_AGENTS = num_agents
STATE_SIZE = int(state_size)  # this environment has 3 time frames stacked together as its state
ACTION_SIZE = action_size
NOISE_AMPLIFICATION = 1
NOISE_AMPLIFICATION_DECAY = 1


# Environment Information
NUM_AGENTS = 2
STATE_SIZE = 24
ACTION_SIZE = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")