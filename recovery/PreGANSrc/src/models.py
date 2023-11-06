import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

## FPE
class FPE_16(nn.Module):
	def __init__(self):
		super(FPE_16, self).__init__()
		self.name = 'FPE_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
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

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_16(nn.Module):
	def __init__(self):
		super(Gen_16, self).__init__()
		self.name = 'Gen_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_16(nn.Module):
	def __init__(self):
		super(Disc_16, self).__init__()
		self.name = 'Disc_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


## FPE
class FPE_50(nn.Module):
	def __init__(self):
		super(FPE_50, self).__init__()
		self.name = 'FPE_50'
		self.lr = 0.0001
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 50
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
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

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
	def __init__(self):
		super(Attention_50, self).__init__()
		self.name = 'Attention_50'
		self.lr = 0.0008
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		# self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats), 
		# 		nn.Sigmoid())	for i in range(1)]
		# self.encoder_atts = nn.ModuleList(self.atts)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		# for at in self.encoder_atts:
		# 	inp = torch.cat((t.view(-1), s.view(-1)))
		# 	ats = at(inp).reshape(self.n_feats, self.n_feats)
		# 	t = torch.matmul(t, ats)	
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

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
	def __init__(self):
		super(Gen_50, self).__init__()
		self.name = 'Gen_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
	def __init__(self):
		super(Disc_50, self).__init__()
		self.name = 'Disc_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


############## PreGANPlus Models ##############

# Transformer Model
class Transformer_16(nn.Module):
	def __init__(self):
		super(Transformer_16, self).__init__()
		self.name = 'Transformer_16'
		self.lr = 0.0001
		self.n_hosts = 16
		feats = 3 * self.n_hosts
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.time_encoder = nn.Sequential(
			nn.Linear(feats, feats * 2 + 1), 
		)
		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.encoder = TransformerEncoder(encoder_layers, 1)
		a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
		self.anomaly_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts), 
		)
		self.softm = nn.Softmax(dim=1)
		p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
		self.prototype_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, PROTO_DIM * self.n_hosts), 
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		t = torch.squeeze(t, 1)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		o = torch.cat((t, gat_t), dim=1)
		t = o * math.sqrt(self.n_feats)
		t = self.pos_encoder(t) # window size, batch size (1), feats (3 metrics * 16 hosts)
		memory = self.encoder(t)	
		return memory

	def anomaly_decode(self, t, memory):
		anomaly_scores = self.anomaly_decoder(t, memory)
		anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
		return anomaly_scores

	def prototype_decode(self, t, memory):
		prototypes = self.prototype_decoder(t, memory)
		prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
		return prototypes

	def forward(self, t, s):
		encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
		t = t.unsqueeze(dim=1)
		memory = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(encoded_t, memory)
		prototypes = self.prototype_decode(encoded_t, memory)
		return anomaly_scores, prototypes