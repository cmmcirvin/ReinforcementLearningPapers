# Prioritized Experience Replay
# Central idea - instead of randomly sampling from past experiences, we can prioritize some transitions that the agent can learn more from.
# These high-valued transitions are sampled more frequently as a function of their TD error
# TD error delta indicates how unexpected a particular transition is, that is, how far the error is from the bootstrapped estimate

from argparse import ArgumentParser
import gymnasium as gym
from collections import namedtuple
import heapq
import torch
import torch.nn as nn
import numpy as np
from operator import itemgetter
#from tqdm import tqdm

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'discount', 'next_state'])

class PrioritizedMemory:
    def __init__(self, capacity):
        self.heap = []
        self.capacity = capacity
    def push(self, priority, transition):
        if self.capacity <= len(self.heap):
            self.heap = self.heap[:-1]
        heapq.heappush(self.heap, (priority, transition))

    def probabilities(self):
        priorities = [item[0] for item in self.heap]
        total_priorities = sum(priorities)
        return [priority / total_priorities for priority in priorities]

    def sample(self, num_samples):
        probs = self.probabilities()
        idxes = np.random.choice(np.arange(len(probs)), size=num_samples, replace=False, p=probs)
        probs = itemgetter(*idxes)(probs)
        transitions = itemgetter(*idxes)(self.heap)
        return transitions, probs, idxes

    def get_max_priority(self):
        if len(self.heap) == 0:
            return 1
        return max(self.heap, key=itemgetter(0))[0]

    def __getitem__(self, idx):
        return self.heap[idx]

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(512, out_dim),
            nn.Dropout(0.2),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.softmax(x)
        return x

def encode_state(state_idx):
    state = torch.zeros(48)
    state[state_idx] = 1
    return state


def get_action(net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)

    return torch.argmax(net(encode_state(state))).item()

#def td_error_loss(net, target_net

def run(args):
    env = gym.make("CliffWalking-v0") # Environment for the agent to explore
    prev_state, _ = env.reset()

    delta = 0

    # Initialize policy and target networks
    net = DQN(in_dim=48, out_dim=4)
    target_net = DQN(in_dim=48, out_dim=4)
    target_net.load_state_dict(net.state_dict())
    
    # Create prioritized memory
    H = PrioritizedMemory(args.replay_size)

    # Iterate for the number of timesteps specified in the arguments
    for step in range(1, args.budget):

        # Select an action based on the policy
        action = get_action(net, prev_state, args.epsilon)

        # Take the action in the environment
        state, reward, terminated, truncated, _ = env.step(action)

        # Store the transition
        p_t = H.get_max_priority()
        tr = Transition(prev_state, action, reward, args.gamma, state)
        H.push(p_t, tr)

        # Update net weights
        if step % args.replay_period == 0:

            # Perform batch_size update steps
            transitions, probs, idxes = H.sample(args.batch_size)

            # Create tensors for previous states and states from transitions
            states = torch.vstack([encode_state(x[1].state) for x in transitions])
            next_states = torch.vstack([encode_state(x[1].next_state) for x in transitions])

            # Calculate the maximum importance-sampling weight from the distribution
            max_w = (args.replay_size * max(H.probabilities())) ** (-1 * args.beta)
            max_w = torch.tensor([max_w] * args.batch_size)

            # Calculate the importance-sampling weights for each transition 
            importance_weights = torch.tensor([args.replay_size] * args.batch_size) * torch.tensor(list(probs))
            importance_weights = torch.pow(importance_weights, (-1 * args.beta)) / max_w

            # Calculate the TD-error for the action
            td_errors = transition.reward + transition.gamma * target_net(transition.state)[torch.argmax(net(transition.state))] - net(transition.prev_state)[transition.action]

            # Update the transition priorities back into the heap with an updated priority 
#            H.push(td_errors, transition)

            # Add the weight update
#            delta += w_j * delta_j * 





                 







        
        



        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=8) # Number of transitions in one update pass
    parser.add_argument('-mu', '--step_size', type=float, default=0.25) # Measure of how much to update model weights
    parser.add_argument('-K', '--replay_period', type=int, default=16) # How frequently to update model weights
    parser.add_argument('-N', '--replay_size', type=int, default=1000) # Size of the replay memory (number of transitions stored)
    parser.add_argument('-a', '--alpha', type=float, default=0.7) # How much prioritization to use (0 => no prioritization)
    parser.add_argument('-b', '--beta', type=float, default=0.5) # Importance sampling weight (tool for reducing variance)
    parser.add_argument('-t', '--budget', type=int, default=1000) # Number of timesteps
    parser.add_argument('-g', '--gamma', type=float, default=0.7) # Discount factor
    parser.add_argument('-e', '--epsilon', type=float, default=0.0) # How often to take a random action
    parser.add_argument('-p', '--prioritized_replay', action='store_true') # Whether or not to use a prioritized replay memory
    args = parser.parse_args()

    run(args)
