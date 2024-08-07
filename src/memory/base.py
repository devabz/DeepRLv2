import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @property
    def prioritized(self):
        return False
    
    @abstractmethod
    def append(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def sample(self, *args, **kwargs) -> tuple:
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass