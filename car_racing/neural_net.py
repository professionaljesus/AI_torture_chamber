import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, image_shape, n_output):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(5712, 120)
        self.fc2 = nn.Linear(5712, 256)
        self.fc3 = nn.Linear(256, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

