import torch
import numpy as np
from base import BaseMemory


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def update(indices: np.ndarray, priorities: np.ndarray, buffer_size: int, tree: np.ndarray) -> None:
    tree_indices = indices + buffer_size - 1 
    differences = priorities - tree[tree_indices]
    while np.any(tree_indices >= 0):
        for index, difference in zip(tree_indices, differences):
            tree[index] += difference
        
        parent_indices = (tree_indices - 1) // 2
        tree_indices = parent_indices
    
def sample(priorities: np.ndarray, buffer_size: int, tree: np.ndarray) -> np.ndarray:
    priorities = priorities.copy()
    indicies = np.zeros_like(priorities, dtype=int)
    while any(indicies < buffer_size - 1):
        left_tree_indices = 2* indicies + 1
        right_tree_indices = left_tree_indices + 1
        
        left_batch_indices = np.where(priorities < tree[left_tree_indices])[0]
        right_batch_indices = np.where(priorities >= tree[left_tree_indices])[0]
        
        indicies[left_batch_indices] = left_tree_indices[left_batch_indices]
        indicies[right_batch_indices] = right_tree_indices[right_batch_indices]
        priorities[right_batch_indices] -= tree[left_tree_indices[right_batch_indices]]
        
    return indicies - buffer_size + 1


class PER(BaseMemory):
    def __init__(self, state_dims, action_dims, buffer_size):
        self.s = np.zeros((buffer_size, state_dims))    # state
        self.a = np.zeros((buffer_size, action_dims))   # action
        self.r = np.zeros((buffer_size, 1))             # reward
        self.n = np.zeros((buffer_size, state_dims))    # next_state
        self.d = np.zeros((buffer_size, 1))             # done
        self.t = np.zeros((buffer_size, 1))             # truncate
        self.tree = np.zeros((2*buffer_size - 1, ))     # sum tree
        self.buffer_size = buffer_size
        self._size = 0
        self.clock = 0
        self.samples_given = 0
        
    def append(
        self, 
        steps: np.ndarray, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        next_states: np.ndarray, 
        dones: np.ndarray, 
        truncated: np.ndarray
    ) -> None:
        
        indicies = steps % self.buffer_size
        self.s[indicies] = states
        self.a[indicies] = actions
        self.r[indicies] = rewards
        self.n[indicies] = next_states
        self.d[indicies] = dones
        self.t[indicies] = truncated
        
        self._size = min(self._size + 1, self.buffer_size)
        
    def update(self, indicies: np.ndarray, priorities: np.ndarray) -> None:
        update(
            indices=indicies, 
            priorities=priorities, 
            buffer_size=self.buffer_size, 
            tree=self.tree
        )
    
    def sample(self, batch_size: int) -> tuple:
        indicies = sample(
            priorities=np.random.uniform(0, self.tree[0], batch_size),
            buffer_size=self.buffer_size,
            tree=self.tree
        )
        
        return (
            indicies,
            torch.Tensor(self.s[indicies]).to(DEVICE),
            torch.Tensor(self.a[indicies]).to(DEVICE),
            torch.Tensor(self.r[indicies]).to(DEVICE),
            torch.Tensor(self.n[indicies]).to(DEVICE),
            torch.Tensor(self.d[indicies]).to(DEVICE),
            torch.Tensor(self.t[indicies]).to(DEVICE)
        )
    
    def __len__(self):
        return self._size
    

if __name__ == "__main__": 
    import gymnasium as gym
    from tqdm import tqdm
    
    env = gym.make('HalfCheetah-v4')
    
    buffer_size = 1_000
    batch_size = 512
    train_step = 50
    total_steps = buffer_size + 10_000
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    
    per = PER(state_dims, action_dims, buffer_size)
    state, _ = env.reset()
    done, truncated = False, False
    progress_bar = tqdm(range(total_steps), total=total_steps)
    batch = []
    
    # Repeat for N steps
    for step in progress_bar:
        progress_bar.set_description_str(str(dict(index=step%buffer_size, memory=len(per), batch=len(batch), buffer_size=buffer_size, train_step=train_step)))
        
        # Perform action
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store transition
        per.append(
            steps=step, 
            states=state, 
            actions=action, 
            rewards=reward, 
            next_states=next_state,
            dones=done,
            truncated=truncated 
        )
        
        if len(per) >= batch_size and step % train_step == 0:
            # Sample batch
            indices, *batch = per.sample(batch_size=batch_size)
            
            # Update network
            ...
            
         
        if done or truncated:
            state, _ = env.reset()