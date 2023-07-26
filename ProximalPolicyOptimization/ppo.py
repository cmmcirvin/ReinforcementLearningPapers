import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from operator import attrgetter
import sys
sys.path.append("..")
from DeepQNetwork.DQN import DQN
from utils import default_args, Memory, Transition 

def run(args):
    SummaryWriter()
    mem = Memory(args.capacity)

    env = gym.make("LunarLander-v2", render_mode="human")
    prev_state, _ = env.reset()
    prev_state = torch.tensor(prev_state)
    num_states = prev_state.shape[0]
    num_actions = env.action_space.n

    policy = DQN(num_states, num_actions)
    target = DQN(num_states, num_actions)
    target.load_state_dict(policy.state_dict())

    nn.L1Loss()
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3) 

    tr_elems = ['state', 'action', 'reward', 'discount', 'next_state']
    getters = [attrgetter(elem) for elem in tr_elems]
    epsilon = args.epsilon_initial

    pbar = tqdm(range(1, args.budget))
    for step in pbar:
        action = policy.take_action(prev_state)
        state, reward, terminated, truncated, _ = env.step(action)
        state = torch.tensor(state)
        mem.push(Transition(prev_state, action, reward, args.gamma, state))
        prev_state = state.clone()

        if step % 4 == 0 and step > 10:
            with torch.enable_grad():
                transitions = mem.sample(10)
                states, actions, rewards, discounts, next_states = [torch.stack([torch.tensor(getters[i](item)) for item in transitions]) for i in range(len(getters))]
    
                target_Qs = rewards + discounts * torch.max(target(next_states), 1).values
                Qs = policy(states)[torch.arange(len(actions)), actions]
                advantage = target_Qs - Qs
                r_t = Qs / target(states)[torch.arange(len(actions)), actions]
                clipped_rt = torch.clip(r_t, 1 - args.clip_weight, 1 + args.clip_weight)
                
                loss = torch.mean(torch.min(r_t * advantage, clipped_rt * advantage))

                optim.zero_grad()
                loss.backward()
                optim.step()

            target.soft_update(policy)
            pbar.set_description(f'Loss: {loss:.3f}, Epsilon: {epsilon:.3f}')

        epsilon *= args.epsilon_decay
        if epsilon <= args.epsilon_final:
            epsilon = args.epsilon_final

        if terminated or truncated:
            prev_state, _ = env.reset()
            prev_state = torch.tensor(prev_state)

if __name__ == "__main__":
    parser = default_args()
    args = parser.parse_args()
    
    run(args)
