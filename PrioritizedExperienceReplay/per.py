# Prioritized Experience Replay
# Central idea - instead of randomly sampling from past experiences, we can prioritize some transitions that the agent can learn more from.
# These high-valued transitions are sampled more frequently as a function of their TD error
# TD error delta indicates how unexpected a particular transition is, that is, how far the error is from the bootstrapped estimate

from argparse import ArgumentParser
import gymnasium as gym
from collections import namedtuple
import heapq

Transition = namedtuple('Transition', ['state, action, reward, discount, next_state'])

class BinaryHeap:
    def __init__(self):
        self.heap = []
    def push(self, transition):
        heapq.heappush(self.heap, transition)
    def pop(self):
        return heapq.heappop(self.heap)
    def __getitem__(self, idx):
        return self.heap[idx]

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = BinaryHeap(capacity)

    def sample(self):
        return self.heap.pop()

#def run()
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-k', '--minibatch_size', type=int, default=8) # Number of transitions in one update pass
    parser.add_argument('-mu', '--step_size', type=float, default=0.25) # Measure of how much to update model weights
    parser.add_argument('-K', '--replay_period', type=int, default=16) # How frequently to update model weights
    parser.add_argument('-N', '--replay_size', type=int, default=1000) # Size of the replay memory (number of transitions stored)
    parser.add_argument('-a', '--alpha', type=float, default=0.7) # How much prioritization to use (0 => no prioritization)
    parser.add_argument('-b', '--beta', type=float, default=0.5) # Importance sampling weight (tool for reducing variance)
    parser.add_argumemt('-t', '--budget', type=int, default=1000) # Number of timesteps
    parser.add_argument('-p', '--prioritized_replay', action='store_true')
    parser.parse_args()

    bh = BinaryHeap(100)

#    env = gym.make("CliffWalking-v0") # Environment for the agent to explore
#    if parser.prioritized_replay:
#        H = PrioritizedMemory() # Use prioritized replay memory 
#    else:
#        H = RandomMemory() # Use random replay memory

    #run()
