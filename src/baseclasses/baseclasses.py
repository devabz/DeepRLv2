import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @abstractmethod
    def append(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def sample(self, *args, **kwargs):
        ...


class BaseAgent(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    @abstractmethod
    def update_offline(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def update_online(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def select_action(self, state: np.ndarray, greedy: bool = False):
        ...
    
