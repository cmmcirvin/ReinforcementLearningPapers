import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import copy
from collections import namedtuple
from tqdm import tqdm
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'discount', 'next_state'])

# Simple DQN that can be built off of for implementing more complex networks
class DQN(nn.Module):
    
    def __init__(self, env):
        super(DQN, self).__init__()
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.net = nn.Sequential(
            nn.Linear(self.num_states, self.num_actions, bias=False),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

    def take_action(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return torch.randint(self.num_actions, (1,)).item()
        else:
            return self.net(state).argmax().item()
