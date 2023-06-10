import torch 
from abc import ABC, abstractmethod

class Component(torch.nn.Module, ABC):
    def __init__(self, device):
        super(Component, self).__init__()
        self.device = device
    
    @abstractmethod        
    def designModelArch(self):
        pass
