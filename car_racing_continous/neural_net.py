import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torchrl.modules.distributions import TruncatedNormal

class ActorCriticCNN(nn.Module):
    def __init__(self, n_outputs, std=0.0):
        super().__init__()

        self.crit_conv1 = nn.Conv2d(2, 16, 5)
        self.crit_conv2 = nn.Conv2d(16, 8, 5)
        self.crit_fc1 = nn.Linear(392, 1)

        self.act_conv1 = nn.Conv2d(2, 16, 5)
        self.act_conv2 = nn.Conv2d(16, 8, 5)
        self.act_fc1 = nn.Linear(392, n_outputs)

        self.log_std = nn.Parameter(torch.ones(1, n_outputs) * std)


    def forward(self, x):
        x_copy = x.detach().clone()
        x = F.max_pool2d(F.relu(self.crit_conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.crit_conv2(x)), 2, 2)
        x = torch.flatten(x, 1)
        value = F.relu(self.crit_fc1(x))

        x = F.max_pool2d(F.relu(self.act_conv1(x_copy)), 2, 2)
        x = F.max_pool2d(F.relu(self.act_conv2(x)), 2, 2)
        x = torch.flatten(x, 1)
        mu = F.tanh(self.act_fc1(x))


        std   = self.log_std.exp().expand_as(mu)
        dist  = TruncatedNormal(mu, std)
        return dist, value
