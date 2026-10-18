import torch
import torch.optim as optim
from torch.utils.data import DataLoader
class ClassifyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        
