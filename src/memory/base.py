import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @abstractmethod
    def append(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def sample(self, *args, **kwargs) -> tuple:
        pass