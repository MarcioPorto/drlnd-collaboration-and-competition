# DRLND Project 3 Report (Collaboration and Competition)

## Introduction

For this project, I decided to implement the MADDPG algorithm, inspired by the implementation used in the DRLND Nanodegree Lab for the physical deception environment. MADDG is a great choice for this environment due to the existence of multiple agents.

I also added a `DDPG_Tennis.ipynb` solution notebook using the DDPG algorithm I implemented in the previous project for this nanodegree. The reason DDPG works so well in this here, without any changes from my solution to project 2 is that the two agents in the `Tennis` environment are trying to learn the same exact policy and have the same reward structure.

## Learning Algorithm

The [MAPPDG algorithm](https://arxiv.org/pdf/1706.02275.pdf) was introduced as an extension of DDPG for multi-agent environments. One can think of MADDPG as a wrapper for handling multiple DDPG agents. But the power of this algorithm is that it adopts a framework of centralized training and decentralized execution, where extra information used during training is not used during testing. More specifically, the training process makes use of both actors and critics, just like DDPG. The difference is that the input to each agent's critic network consists of all the observations and actions for all the agents combined. However, since only the actor is present during testing, that extra information used during training effectively goes away. This framework makes MADDPG flexible enough to handle competitive, collaborative and mixed environments.

Because of the similarities with DDPG, the hyperparamters used in MADDPG are very similar, and are listed below:

```
BUFFER_SIZE                 # replay buffer size
BATCH_SIZE                  # minibatch size
GAMMA                       # discount factor
TAU                         # for soft update of target parameters
LR_ACTOR                    # learning rate of the actor 
LR_CRITIC                   # learning rate of the critic
WEIGHT_DECAY                # L2 weight decay
UPDATE_EVERY                # weight update frequency
NOISE_AMPLIFICATION         # exploration noise amplification
NOISE_AMPLIFICATION_DECAY   # noise amplification decay
```

The last two hyperparameters were added to amplify the exploration noise used by the agent in an effort to learn the optimal policy faster.

In MADDPG, each agent has its own actor local, actor target, critic local and critic target networks. The model architecture for MADDPG resembles DDPG very closely. The difference is that I found better results when concatenating the states and actions as the input to the critic's first hidden layer, whereas DDPG concatenates the actions to the input of the second hidden layer.

## Training and Results

Training the DDPG solution was very simple. Only a few changes in the hyperparameters from Project 2 solved the environment for me:

![DDPG Plot of Rewards](https://github.com/MarcioPorto/drlnd-collaboration-and-competition/blob/master/ddpg_tennis_training.png)

As you can see above, the training was a little unstable, and the agent never achieved a rolling score above the 0.5 required to solve this environment. I believe I could have stabilized this result with more time, but my main focus in this project was to get MADDPG working.

I found training to be particularly challenging for the MADDPG implementation. Initially, I had issues getting the full implementation to work as expected. After that was worked out, I found the training process to be very sensitive to the hyperparameters. It took a good amount of trial and error to finally get to something that worked:

![MADDPG Plot of Rewards](https://github.com/MarcioPorto/drlnd-collaboration-and-competition/blob/master/maddpg_tennis_training.png)

## Future Work Ideas

I would like to spend more time playing with the hyperparameters used to see if I can get the agents to achieve the target score for this environment a little faster.

Next, I would like to use this implementation to take a crack at the `Soccer` Unity environment, as it presents a more challenging environment where agents can have different reward structures.