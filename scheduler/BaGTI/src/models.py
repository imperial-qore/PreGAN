import torch
import torch.nn as nn
from .constants import *
from .npn import *

from sys import argv

class energy_latency_16(nn.Module):
    def __init__(self):
        super(energy_latency_16, self).__init__()
        self.name = "energy_latency_16"
        self.find = nn.Sequential(
            nn.Linear(16 * 18, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
