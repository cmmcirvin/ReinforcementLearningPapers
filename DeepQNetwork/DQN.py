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
    
    def __init__(self, num_states, num_actions, gamma, optim, lr):
        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        self.net = nn.Sequential(
#            nn.Linear(self.num_states, 48, bias=False),
#            nn.ReLU(),
#            nn.Linear(48, 48, bias=False),
#            nn.ReLU(),
#            nn.Linear(48, self.num_actions, bias=False)
            nn.Linear(self.num_states, self.num_actions, bias=False),
            nn.ReLU()
        )

        self.optim = optim(self.net.parameters(), lr=lr)
        self.target_net = copy.deepcopy(self.net)

    def take_action(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return torch.randn(self.num_actions).item()
        else:
            return self.net(F.one_hot(torch.tensor(state), self.num_states).to(torch.float32)).argmax().item()

    def forward(self, x):
        return self.net(x)

    def update_weights(self, batch):
        priorities = [x[0] for x in batch]
        transitions = [y[1] for y in batch]
        states, actions, rewards, discounts, next_states = torch.tensor(transitions).hsplit(5)

        states = F.one_hot(states.flatten().to(torch.long), self.num_states).to(torch.float32)
        next_states = F.one_hot(next_states.flatten().to(torch.long), self.num_states).to(torch.float32)

        target_Qs = rewards + self.gamma * self.target_net(next_states)
        Qs = self.net(states).gather(1, actions.to(torch.long))
        
        loss = F.mse_loss(Qs, target_Qs)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.target_net.load_state_dict(self.net.state_dict())

        return loss.item()


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    model = DQN(env.observation_space.n, env.action_space.n, 0.7, torch.optim.Adam, 0.001)

    for i in tqdm(range(1000)):
        state, _ = env.reset()
        done = False
        while not done:
            action = torch.argmax(model.net(F.one_hot(torch.tensor(state), model.num_states).to(torch.float32))).item()
            next_state, reward, terminated, truncated, _= env.step(action)
            batch = [Transition(state, action, reward, 0.7, next_state),
                     Transition(state, action, reward, 0.7, next_state),
                     Transition(state, action, reward, 0.7, next_state),
                     Transition(state, action, reward, 0.7, next_state)]
            model.update_weights(batch)
            state = next_state





