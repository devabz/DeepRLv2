import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.updates = 0
        
    @abstractmethod
    def update_offline(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def update_online(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, greedy: bool = False):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass