import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class ClassifyDataSet(Dataset):
    def __init__(self, images, labels, classes, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = labels
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        if  self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
