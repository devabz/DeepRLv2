import os
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from src.agents.backend import ActorGaussian, CriticNetwork, ValueNetwork
from src.agents.base import BaseAgent
from src.memory.base import BaseMemory

class  SAC(BaseAgent):
    def __init__(
            self,  
            state_dims: int, 
            action_dims: int, 
            hidden_dims: int,  
            reparam_noise: float,
            lr_a: float, 
            lr_v: float, 
            lr_c: float, 
            tau: float,
            gamma: float,  
            min_action: np.ndarray,
            max_action: np.ndarray,  
            memory: BaseMemory, 
            reward_scale: int = 1,
            DEVICE: str = None
        ) -> None:
        
        super().__init__()
        self.DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE is None else DEVICE
        
        self.critic = CriticNetwork(
            state_dims,
            action_dims,
            hidden_dims
        ).to(self.DEVICE)
        
        self.value = ValueNetwork(
            state_dims,
            hidden_dims
        ).to(self.DEVICE)
        
        self.value_target = deepcopy(self.value)
        
        self.actor = ActorGaussian(
            state_dims, action_dims, hidden_dims, 
            min_action=torch.Tensor(min_action).to(self.DEVICE), 
            max_action=torch.Tensor(max_action).to(self.DEVICE),
            reparam_noise=reparam_noise
        ).to(self.DEVICE)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_v)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        
        self.tau = tau
        self.gamma = gamma
        self.memory = memory
        self.min = min_action
        self.max = max_action
        self.reward_scale = reward_scale


    def update_offline(self, *args, **kwargs):
        return super().update_offline(*args, **kwargs)
    
    def _update_networks(self, *batch):
        self.updates += 1
        logs = defaultdict(lambda : None)
        
        states, actions, rewards, next_states, dones, truncated = batch
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_values = torch.min(*self.critic_target(next_states, next_actions))
            target = rewards + q_values * self.gamma * (1 - dones)
    
    def update_online(self, *args, **kwargs):
        logs = dict()
        s = kwargs['state']
        a = kwargs['action']
        r = kwargs['reward']
        n = kwargs['next_state']
        d = kwargs['done']
        t = kwargs['truncated']
        step = kwargs['step']
        steps = kwargs['steps']
        episode = kwargs['episode']
        batch_size = kwargs['batch_size']
        update_freq = kwargs['update_freq']
        
        
        self.memory.append(steps=step, states=s, actions=a, rewards=r, next_states=n, dones=d, truncated=t)
        
        if len(self.memory) >= batch_size and step % update_freq == 0:
            indices, *batch = self.memory.sample(batch_size=batch_size)
            logs = self._update_networks(*batch)
            if self.memory.prioritized:
                priorities = np.abs(logs['td_errors'])
                self.memory.update(indices, priorities)
        
        return logs
    
    
    def select_action(self, state: np.ndarray, greedy: bool = False):
        with torch.no_grad():
            state = torch.Tensor(state).to(self.DEVICE)
            action, _ = self.actor.sample(state, reparameterize=False)
        
        return action.cpu().detach().numpy()
    
    def save(self, path):
        torch.save(self.critic.to('cpu').state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.actor.to('cpu').state_dict(), os.path.join(path, 'actor.pth'))

        self.critic.to(self.DEVICE)
        self.actor.to(self.DEVICE)
        
    def load(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), weights_only=True))
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), weights_only=True))
    
    def _soft_update(self, main, target):
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        
        