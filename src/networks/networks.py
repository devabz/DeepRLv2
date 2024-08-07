import torch
import torch.nn as nn
from torch.nn.functional import relu, tanh


class ValueNetwork(nn.Module):
    def __init__(self, state_dims: int, hidden_dims: int):
        super().__init__()
        self.c1 = nn.Linear(state_dims, hidden_dims)
        self.c2 = nn.Linear(hidden_dims, hidden_dims)
        self.c3 = nn.Linear(hidden_dims, 1)
        
    def forward(self, states: torch.Tensor):
        x = relu(self.c1(states))
        x = relu(self.c2(x))
        x = self.c3(x)

        return x
    
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, hidden_dims: int):
        super().__init__()
        self.c11 = nn.Linear((state_dims + action_dims), hidden_dims)
        self.c12 = nn.Linear(hidden_dims, hidden_dims)
        self.c13 = nn.Linear(hidden_dims, 1)
        
        self.c21 = nn.Linear((state_dims + action_dims), hidden_dims)
        self.c22 = nn.Linear(hidden_dims, hidden_dims)
        self.c23 = nn.Linear(hidden_dims, 1)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        sa = torch.cat([states, actions], 1)
        x = relu(self.c11(sa))
        x = relu(self.c12(x))
        x = self.c13(x)

        y = relu(self.c21(sa))
        y = relu(self.c22(y))
        y = self.c23(y)
        
        return x, y
    
    def q1(self, states: torch.Tensor, actions: torch.Tensor):
        sa = torch.cat([states, actions], 1)
        x = relu(self.c11(sa))
        x = relu(self.c12(x))
        x = self.c13(x)

        return x


class ActorNetwork(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, hidden_dims: int, min_action: torch.Tensor, max_action: torch.Tensor):
        super().__init__()
        self.c1 = nn.Linear(state_dims, hidden_dims)
        self.c2 = nn.Linear(hidden_dims, hidden_dims)
        self.c3 = nn.Linear(hidden_dims, action_dims)
        
        self.max = max_action
        self.min = min_action
        
    def forward(self, states: torch.Tensor):
        x = relu(self.c1(states))
        x = relu(self.c2(x))
        x = tanh(self.c3(x)) * self.max
        x = x.clamp(self.min, self.max)
        return x
    

if __name__ == "__main__":
    import numpy as np
    import gymnasium as gym
    
    # Collect arguments
    env = gym.make('Pusher-v4')
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    hidden_dims = 128
    sample_size = 256
    
    max_action = torch.Tensor(env.action_space.high)
    min_action = torch.Tensor(env.action_space.low)
    
    # Sample states
    states = torch.Tensor(np.random.random(size=(sample_size, state_dims)))
    
    # Initialize networks
    value = ValueNetwork(state_dims, hidden_dims)
    critic = CriticNetwork(state_dims, action_dims, hidden_dims)
    actor = ActorNetwork(state_dims, action_dims, hidden_dims, min_action, max_action)
    
    # Perform inference
    actions = actor(states)
    values = value(states)
    q1, q2 = critic(states, actions)
    
    # Results
    print(f'Actions: {actions.shape} V: {values.shape} Q1: {q1.shape} Q2: {q2.shape}')