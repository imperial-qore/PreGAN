import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

## Simple FCN Model
class FCN_16(nn.Module):
	def __init__(self):
		super(FCN_16, self).__init__()
		self.name = 'FCN_16'
		self.lr = 0.001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 1 
		self.n_hidden = 32
		self.n = self.n_window * self.n_feats
		self.fcn = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, t):
		t = self.fcn(t.view(-1))
		return t.view(self.n_window, self.n_feats)
