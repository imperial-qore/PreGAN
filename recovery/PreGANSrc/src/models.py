import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
	def __init__(self):
		super(Attention_50, self).__init__()
		self.name = 'Attention_50'
		self.lr = 0.001
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 5 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats), 
				nn.ReLU(True))	for i in range(1)]
		self.encoder_atts = nn.ModuleList(self.atts)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.ReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.zeros(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		for at in self.encoder_atts:
			inp = torch.cat((t.view(-1), s.view(-1)))
			ats = at(inp).reshape(self.n_feats, self.n_feats)
			t = torch.matmul(t, ats)	
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes