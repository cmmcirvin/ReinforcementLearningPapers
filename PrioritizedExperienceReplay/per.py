# Prioritized Experience Replay
# Central idea - instead of randomly sampling from past experiences, we can prioritize some transitions that the agent can learn more from.
# These high-valued transitions are sampled more frequently as a function of their TD error
# TD error delta indicates how unexpected a particular transition is, that is, how far the error is from the bootstrapped estimate

from argparse import ArgumentParser
import gymnasium as gym
from dataclasses import dataclass
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from operator import itemgetter, attrgetter
from tqdm import tqdm
import random
import sys
sys.path.append("..")
from DeepQNetwork.DQN import DQN

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: int
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

class PrioritizedMemory:
    def __init__(self, capacity, beta):
        self.memory = []
        self.capacity = capacity
        self.beta = beta

    def push(self, transition, priority):
        if self.capacity <= len(self.memory):
            random_idx = np.random.randint(0, len(self.memory))
            self.memory.pop(random_idx)
        self.memory.append((priority, transition))

    def priorities(self):
        return [item[0] for item in self.memory]

    def probabilities(self):
        priorities = self.priorities()
        total_priorities = sum(priorities)
        return [priority / total_priorities for priority in priorities]

    def sample(self, batch_size):
        probs = self.probabilities()
        idxes = np.random.choice(np.arange(len(self.memory)), size=batch_size, replace=False, p=probs)
        probs = itemgetter(*idxes)(probs)
        imps = torch.pow(len(self.memory) * torch.tensor(probs), -1 * self.beta)
        imps = imps / max(imps)
        transitions = itemgetter(*idxes)(self.memory)
        return transitions, imps, idxes

    def update(self, priorities, idxes):
        for i, memory_idx in enumerate(idxes):
            self.memory[memory_idx] = (priorities[i].item(), self.memory[memory_idx][1])

    def get_max_priority(self):
        if len(self.memory) == 0:
            return 1
        return max(self.priorities())

    def __getitem__(self, idx):
        return self.memory[idx]

def visualize_q_values(args, net, env):
    # Visualize the Q-values for every state-action pair
    q_values = [] 
    for i in range(4):
        for j in range(12):
            print(net.take_action(12 * i + j, 0.0), end=" ")
        print()

    for state in range(48):
        print(state, net.net(F.one_hot(torch.tensor(state), 48).to(torch.float32)))

def encode_state(state, num_states):
    return F.one_hot(torch.tensor(state), num_states).to(torch.float32)

def run(args):
    env = gym.make("CartPole-v1", render_mode=args.render_mode) # Environment for the agent to explore

    prev_state, _ = env.reset()
    num_states = prev_state.shape[0] # Number of states the agent can be in
    num_actions = env.action_space.n # Number of actions the agent can take

    delta = 0
    epsilon = args.epsilon_initial
    gamma = args.gamma

    # Initialize policy and target networks
    policy = DQN(num_states, num_actions)
    target = DQN(num_states, num_actions)
    target.load_state_dict(policy.state_dict())

    # Initialize optimizer
    optim = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    # Create prioritized memory
    H = PrioritizedMemory(args.replay_size, args.beta)
#    H = Memory(args.replay_size)

    # Set progress bar to iterate over
    pbar = tqdm(range(1, args.budget))
    terminated, truncated = False, False

    writer = SummaryWriter()
    ep_step = 0
    ep = 0

    tr_elems = ['state', 'action', 'reward', 'discount', 'next_state']
    getters = [attrgetter(elem) for elem in tr_elems]

    # Iterate for the number of timesteps specified in the arguments
    for step in pbar:

        with torch.no_grad():
            
            # Select an action based on the policy
            action = policy.take_action(torch.tensor(prev_state), epsilon)

            # Take the action in the environment
            state, reward, terminated, truncated, _ = env.step(action)
            ep_step += 1

            # If the agent reaches the goal
            if terminated:
                # Add additional reward
                reward += args.goal_reward
                terminated = True

            # Store the transition
            tr = Transition(torch.tensor(prev_state), action, reward, gamma, torch.tensor(state))
            H.push(tr, H.get_max_priority())

            # Update the previous state value to store the current state
            prev_state = state

            # If the episode is over, reset the environment
            if terminated or truncated:
                prev_state, _ = env.reset()
                writer.add_scalar("Episode Reward", ep_step, ep)
                ep_step = 0
                ep += 1

        # Update net weights
        if step > args.batch_size and step > args.warmup_steps:

            with torch.enable_grad():

                # Perform batch_size update steps
                transitions, imps, idxes = H.sample(args.batch_size)

                states, actions, rewards, discounts, next_states = [torch.stack([torch.tensor(getters[i](item[1])) for item in transitions]) for i in range(len(getters))]
        
                target_Qs = rewards + discounts * torch.max(target(next_states), 1).values
                Qs = policy(states)[torch.arange(len(actions)), actions]
                
                td_errors = torch.abs(Qs - target_Qs)
                loss = torch.mean(imps * torch.pow(Qs - target_Qs, 2))
        
                optim.zero_grad()
                loss.backward()
                optim.step()

                target.soft_update(policy)

                # Update the transition prorities to the td_errors
                # Note: we use the absolute value of the td_errors and take the negative as heapq uses a minheap by default 
                H.update(td_errors.clone().detach(), idxes)
                
                # Update the progress bar description
                pbar.set_description(f'Loss: {loss:.3f}, Epsilon: {epsilon:.3f}')

            # Update epsilon to a lower value
            epsilon *= args.epsilon_decay
            if epsilon <= args.epsilon_final:
                epsilon = args.epsilon_final

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=16) # Number of transitions in one update pass
    parser.add_argument('-mu', '--step_size', type=float, default=0.25) # Measure of how much to update model weights
    parser.add_argument('-w', '--warmup_steps', type=int, default=256) # Number of steps to take before updating model weights
    parser.add_argument('-K', '--replay_period', type=int, default=16) # How frequently to update model weights
    parser.add_argument('-N', '--replay_size', type=int, default=10000) # Size of the replay memory (number of transitions stored)
    parser.add_argument('-a', '--alpha', type=float, default=1) # How much prioritization to use (0 => no prioritization)
    parser.add_argument('-b', '--beta', type=float, default=0.5) # Importance sampling weight (tool for reducing variance)
    parser.add_argument('-t', '--budget', type=int, default=10000) # Number of timesteps
    parser.add_argument('-g', '--gamma', type=float, default=0.999) # Discount factor
    parser.add_argument('-ei', '--epsilon_initial', type=float, default=0.99) # How often to take a random action initially
    parser.add_argument('-d', '--epsilon_decay', type=float, default=0.99) # Decay rate for epsilon 
    parser.add_argument('-ef', '--epsilon_final', type=float, default=0.0) # How often to take a random action finally 
    parser.add_argument('-p', '--prioritized_replay', action='store_true') # Whether or not to use a prioritized replay memory
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3) # Learning rate for the network
    parser.add_argument('-tau', '--soft_update_rate', type=float, default=5e-3) # Soft update rate of the network 
    parser.add_argument('-rm', '--render_mode', type=str, default=None) # How to render the environment
    parser.add_argument('-gr', '--goal_reward', type=int, default=-10) # Reward for reaching the goal state
    parser.add_argument('-gs', '--goal_state', type=int, default=47) # Goal state of the environment 
    args = parser.parse_args()

    while True:
        run(args)
