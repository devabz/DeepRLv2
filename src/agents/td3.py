import os
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from torch.nn.functional import mse_loss

from src.agents.base import BaseAgent
from src.memory.base import BaseMemory
from src.networks import CriticNetwork, ActorNetwork


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TD3(BaseAgent):
    def __init__(
        self, 
        state_dims: int, 
        action_dims: int, 
        hidden_dims: int, 
        lr_a: float, 
        lr_c: float, 
        tau: float,
        gamma: float,
        policy_delay: int,
        noise: float,
        noise_clip: float,
        noise_decay: float,
        min_action: np.ndarray,  
        max_action: np.ndarray, 
        memory: BaseMemory,
        DEVICE: str = None
    ):
        
        super().__init__()
        self.memory = memory
        __DEVICE__ = ('cuda' if torch.cuda.is_available() else 'cpu') 
        self.DEVICE = __DEVICE__ if DEVICE is None else DEVICE
        
        self.critic = CriticNetwork(
            state_dims, action_dims, hidden_dims
        ).to(self.DEVICE)
        
        self.actor = ActorNetwork(
            state_dims, action_dims, hidden_dims, 
            min_action=torch.Tensor(min_action).to(self.DEVICE), 
            max_action=torch.Tensor(max_action).to(self.DEVICE)
        ).to(self.DEVICE)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        
        self.tau = tau
        self.gamma = gamma
        self.max = max_action
        self.min = min_action
        self.policy_delay = policy_delay
        self.noise = noise
        self.noise_clip = noise_clip
        self.noise_decay = noise_decay
        
    def select_action(self, state, greedy: bool = False):
        state = torch.Tensor(state).to(self.DEVICE)
        with torch.no_grad():
            action = self.actor(state)
            action = action.cpu().numpy()
        
        if not greedy:
            action += np.random.normal(0, self.noise * self.max, size=self.max.shape)
            
        action = action.clip(self.min, self.max)
        
        return action
    
    
    def _soft_update(self, main, target):
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def _update_networks(self, *batch):
        self.updates += 1
        logs = defaultdict(lambda : None)
        
        states, actions, rewards, next_states, dones, truncated = batch
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_values = torch.min(*self.critic_target(next_states, next_actions))
            target = rewards + q_values * self.gamma * (1 - dones)
        
        # Move the critic closer to the target
        q1, q2 = self.critic(states, actions)
        critic_loss = mse_loss(q1, target) + mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
            logs['1_actor_loss'] = actor_loss.detach().cpu().numpy()
            
        logs['1_critic_loss'] = critic_loss.detach().cpu().numpy()
        logs['td_errors'] = (mse_loss(torch.cat((q1.detach(), q2.detach()), 1).mean(1).unsqueeze(1), target)).cpu().numpy()
        return logs
        
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
    
    def update_offline(self, *args, **kwargs):
        return 
    
    def save(self, path):
        torch.save(self.critic.to('cpu').state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.actor.to('cpu').state_dict(), os.path.join(path, 'actor.pth'))

        self.critic.to(self.DEVICE)
        self.actor.to(self.DEVICE)
        
    def load(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), weights_only=True))
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), weights_only=True))