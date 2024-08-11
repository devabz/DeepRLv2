import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ActorGaussian(nn.Module):
    
    def __init__(self, state_dims: int, action_dims: int, hidden_dims: int, min_action: torch.Tensor, max_action: torch.Tensor, reparam_noise: float = 1e-6) -> None:
        super().__init__()
        
        self.l1 = nn.Linear(state_dims, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, hidden_dims)
        
        self.mu = nn.Linear(hidden_dims, action_dims)
        self.sigma = nn.Linear(hidden_dims, action_dims)
        
        self.min = min_action
        self.max = max_action
        
        self.reparam_noise = reparam_noise 
        
    def forward(self, states: torch.Tensor):
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        mu = self.mu(x)
        sigma = self.sigma(x).clamp(self.reparam_noise, 1)
        
        return mu, sigma
    
    def sample(self, state, actions=None, reparameterize=True):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)
        if actions is None:
            if reparameterize:
                actions = probs.rsample()
            else:
                actions = probs.sample()
            
            actions = actions.clamp(self.min, self.max)
            
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - actions.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return actions, log_probs
    
        
        
        
if __name__ == "__main__":
    import numpy as np
    import gymnasium as gym
    
    env = gym.make("HalfCheetah-v4")
    
    hidden_dims = 256
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    
    min_action = torch.Tensor(env.action_space.low)
    max_action = torch.Tensor(env.action_space.high)
    
    reparam_noise = 1e-6
    
    actor = ActorGaussian(state_dims, action_dims, hidden_dims, min_action, max_action, reparam_noise=reparam_noise)
    
    
    states = torch.Tensor(np.random.random(size=(20, state_dims)))
    mu, sigma = actor(states)
    actions_, log_probs = actor.sample(states)
    print(mu.shape, sigma.shape, actions_.shape, log_probs.shape)
    