import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import copy
from collections import namedtuple
from tqdm import tqdm
import numpy as np

# Simple DQN that can be built off of for implementing more complex networks
 
class DQN(nn.Module):
    
    def __init__(self, num_states, num_actions, tau=1e-3):
        super(DQN, self).__init__()
        self.num_states = num_states 
        self.num_actions = num_actions 
        self.tau = tau

        self.net = nn.Sequential(
            nn.Linear(self.num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

    def take_action(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return torch.randint(self.num_actions, (1,)).item()
        else:
            return self.net(state).argmax().item()

    def soft_update(self, other):
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data.copy_(param.data * (1.0 - self.tau) + other_param.data * self.tau)

