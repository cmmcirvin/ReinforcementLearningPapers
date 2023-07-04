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
from tqdm import tqdm

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'discount', 'next_state'])

class PrioritizedMemory:
    def __init__(self, capacity):
        self.heap = []
        self.capacity = capacity

    def push(self, priority, transition):
        if self.capacity <= len(self.heap):
            random_idx = np.random.randint(0, len(self.heap))
            self.heap.pop(random_idx)
        heapq.heappush(self.heap, (priority, transition))

    def probabilities(self):
        priorities = [-1 * item[0] for item in self.heap]
        total_priorities = sum(priorities)
        return [priority / total_priorities for priority in priorities]

    def sample(self, num_samples):
        probs = self.probabilities()
        idxes = np.random.choice(np.arange(len(probs)), size=num_samples, replace=False, p=probs)
        probs = itemgetter(*idxes)(probs)
        transitions = itemgetter(*idxes)(self.heap)
        return transitions, probs, idxes

    def update(self, priorities, idxes):
        for i, heap_idx in enumerate(idxes):
            self.heap[heap_idx] = (priorities[i].item(), self.heap[heap_idx][1])
        heapq.heapify(self.heap)

    def get_max_priority(self):
        if len(self.heap) == 0:
            return -1
        # Use a min here as heapq defaults to a min heap
        return min(self.heap, key=itemgetter(0))[0]

    def __getitem__(self, idx):
        return self.heap[idx]

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 16),
#            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(16, 32),
#            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(32, 16),
#            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(16, out_dim),
#            nn.Dropout(0.1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.softmax(x, dim=0)
        return x

def encode_state(state_idx, num_states=16):
    state = torch.zeros(num_states)
    state[state_idx] = 1
    return state


def get_action(net, state, epsilon, num_states):
    if np.random.rand() < epsilon:
        return np.random.randint(4)

    return torch.argmax(net(encode_state(state, num_states))).item()

def td_loss(H, net, target_net, transitions, probs, num_actions, num_states):
    # Create tensors for previous states and states from transitions
    states = torch.vstack([encode_state(x[1].state, num_states) for x in transitions])
    next_states = torch.vstack([encode_state(x[1].next_state, num_states) for x in transitions])
    rewards = torch.vstack([torch.tensor(x[1].reward) for x in transitions])
    discounts = torch.vstack([torch.tensor(x[1].discount) for x in transitions])
    actions = torch.vstack([torch.tensor(x[1].action) for x in transitions])

    # Calculate the maximum importance-sampling weight from the distribution
    max_w = (args.replay_size * max(H.probabilities())) ** (-1 * args.beta)
    max_w = torch.tensor([max_w] * args.batch_size)

    # Calculate the importance-sampling weights for each transition 
    importance_weights = torch.tensor([args.replay_size] * args.batch_size) * torch.tensor(list(probs))
    importance_weights = torch.pow(importance_weights, (-1 * args.beta)) / max_w

    with torch.enable_grad():
        # Calculate the TD-error for the action
        argmax_q_actions = torch.nn.functional.one_hot(torch.stack([torch.argmax(x) for x in net(next_states)]), num_classes=num_actions)
        q_target_states = torch.sum(target_net(next_states) * argmax_q_actions, axis=1).reshape(-1, 1)
        q_prev_states = torch.sum(net(states) * torch.nn.functional.one_hot(actions.reshape(actions.shape[0]), num_classes=num_actions), axis=1).reshape(-1, 1)
        td_errors = rewards + discounts * q_target_states - q_prev_states
        td_errors = td_errors * importance_weights.reshape(*importance_weights.shape, 1)

    return td_errors

#def fill_memory(args, env, H):
#    # Fill the replay memory with transitions for every state-action paichr
#    env.reset()
#    for state in range(env.observation_space.n):
#        for action in range(env.action_space.n):
#            next_state, reward, done, _ = env.step(action)
#            tr = Transition(state, action, reward, args.discount, next_state)
#            H.push(-1.0, tr)
#
#    return H


def run(args):
    env = gym.make("CliffWalking-v0", render_mode=args.render_mode) # Environment for the agent to explore
    num_actions = env.action_space.n # Number of actions the agent can take
    num_states = env.observation_space.n # Number of states the agent can be in
    prev_state, _ = env.reset()

    delta = 0
    epsilon = args.epsilon_initial
    gamma = args.gamma

    # Initialize policy and target networks
    net = DQN(in_dim=num_states, out_dim=num_actions)
    target_net = DQN(in_dim=num_states, out_dim=num_actions)
    target_net.load_state_dict(net.state_dict())

    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # Create prioritized memory
    H = PrioritizedMemory(args.replay_size)

    # Fill the replay memory with transitions for every state-action pair
#    H = fill_memory(args, env, H)

    # Set progress bar to iterate over
    pbar = tqdm(range(1, args.budget))

    # Iterate for the number of timesteps specified in the arguments
    for step in pbar:

        with torch.no_grad():
            # Select an action based on the policy
            action = get_action(net, prev_state, epsilon, num_states)

            # Take the action in the environment
            state, reward, terminated, truncated, _ = env.step(action)

            # If the agent reaches the goal
            if state == args.goal_state:
                # Add additional reward
                reward += args.goal_reward
                terminated = True

            # Store the transition
            p_t = abs(H.get_max_priority())
            tr = Transition(prev_state, action, reward, gamma, state)
            noise = (torch.randn(1) - 0.5) / 10 # Small noise to break ties
            H.push(-1 * (p_t ** args.alpha) + noise.item(), tr)

            # Update the previous state value to store the current state
            prev_state = state

            # If the episode is over, reset the environment
            if terminated or truncated:
                prev_state, _ = env.reset()

        # Update net weights
        if step % args.replay_period == 0 and step > args.batch_size:

            with torch.enable_grad():

                optim.zero_grad()

                # Perform batch_size update steps
                transitions, probs, idxes = H.sample(args.batch_size)

                # Calculate TD errors
                td_errors = td_loss(H, net, target_net, transitions, probs, num_actions, num_states)

                # Update the transition prorities to the td_errors
                # Note: we use the absolute value of the td_errors and take the negative as heapq uses a minheap by default 
                H.update(-1 * torch.abs(td_errors.clone().detach()), idxes)
                
                # Calculate the loss
                loss = torch.sum(td_errors)
                
                # Update the progress bar description
                pbar.set_description(f'Loss: {loss.item():.3f}, Epsilon: {epsilon:.3f}')

                # Update model parameters
                loss.backward()
                optim.step()

                if step % (args.replay_period) == 0:
                    # Update target network
                    target_net.load_state_dict(net.state_dict())

            # Update epsilon to a lower value
            epsilon *= args.epsilon_decay
            if epsilon <= args.epsilon_final:
                epsilon = args.epsilon_final
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=32) # Number of transitions in one update pass
    parser.add_argument('-mu', '--step_size', type=float, default=0.25) # Measure of how much to update model weights
    parser.add_argument('-K', '--replay_period', type=int, default=16) # How frequently to update model weights
    parser.add_argument('-N', '--replay_size', type=int, default=100) # Size of the replay memory (number of transitions stored)
    parser.add_argument('-a', '--alpha', type=float, default=1) # How much prioritization to use (0 => no prioritization)
    parser.add_argument('-b', '--beta', type=float, default=0.5) # Importance sampling weight (tool for reducing variance)
    parser.add_argument('-t', '--budget', type=int, default=10000) # Number of timesteps
    parser.add_argument('-g', '--gamma', type=float, default=0.7) # Discount factor
    parser.add_argument('-ei', '--epsilon_initial', type=float, default=0.9) # How often to take a random action initially
    parser.add_argument('-d', '--epsilon_decay', type=float, default=0.99) # Decay rate for epsilon 
    parser.add_argument('-ef', '--epsilon_final', type=float, default=0.1) # How often to take a random action finally 
    parser.add_argument('-p', '--prioritized_replay', action='store_true') # Whether or not to use a prioritized replay memory
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1) # Learning rate for the network
    parser.add_argument('-rm', '--render_mode', type=str, default=None) # How to render the environment
    parser.add_argument('-gr', '--goal_reward', type=int, default=1000) # Reward for reaching the goal state
    parser.add_argument('-gs', '--goal_state', type=int, default=47) # Goal state of the environment 
    args = parser.parse_args()

    while True:
        run(args)
