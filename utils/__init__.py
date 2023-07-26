from dataclasses import dataclass
from argparse import ArgumentParser
import random
import torch

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float 
    discount: float
    next_state: torch.Tensor

class Memory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

    def push(self, transition):
        if self.capacity <= len(self.memory):
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def default_args():   
    parser = ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=16) # Number of transitions in one update pass
    parser.add_argument('-mu', '--step_size', type=float, default=0.25) # Measure of how much to update model weights
    parser.add_argument('-w', '--warmup_steps', type=int, default=256) # Number of steps to take before updating model weights
    parser.add_argument('-K', '--replay_period', type=int, default=16) # How frequently to update model weights
    parser.add_argument('-N', '--replay_size', type=int, default=10000) # Size of the replay memory (number of transitions stored)
    parser.add_argument('-a', '--alpha', type=float, default=0.7) # How much prioritization to use (0 => no prioritization)
    parser.add_argument('-b', '--beta', type=float, default=0.5) # Importance sampling weight (tool for reducing variance)
    parser.add_argument('-t', '--budget', type=int, default=10000) # Number of timesteps
    parser.add_argument('-c', '--capacity', type=int, default=1000) # Capacity of the replay memory
    parser.add_argument('-g', '--gamma', type=float, default=0.999) # Discount factor
    parser.add_argument('-cw', '--clip_weight', type=float, default=0.2) # Clip weight
    parser.add_argument('-ei', '--epsilon_initial', type=float, default=0.99) # How often to take a random action initially
    parser.add_argument('-d', '--epsilon_decay', type=float, default=0.99) # Decay rate for epsilon 
    parser.add_argument('-ef', '--epsilon_final', type=float, default=0.0) # How often to take a random action finally 
    parser.add_argument('-p', '--prioritized_replay', action='store_true') # Whether or not to use a prioritized replay memory
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3) # Learning rate for the network
    parser.add_argument('-tau', '--soft_update_rate', type=float, default=5e-3) # Soft update rate of the network 
    parser.add_argument('-rm', '--render_mode', type=str, default=None) # How to render the environment
    parser.add_argument('-gr', '--goal_reward', type=int, default=-10) # Reward for reaching the goal state
    parser.add_argument('-gs', '--goal_state', type=int, default=47) # Goal state of the environment 
    return parser

