import numpy as np
from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *
from metrics.powermodels.PMRaspberryPi import *
from metrics.powermodels.PMRaspberryPi4B import *
from metrics.powermodels.PMRaspberryPi4B8G import *
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *
from metrics.powermodels.PMXeon_X5570 import *

class RPiEdge():
	def __init__(self, num_hosts):
		self.num_hosts = num_hosts
		self.edge_hosts = round(num_hosts * 2)
		self.types = {
			'RPi4B':
				{
					'IPS': 4029,
					'RAMSize': 4295,
					'RAMRead': 372.0,
					'RAMWrite': 200.0,
					'DiskSize': 32212,
					'DiskRead': 13.42,
					'DiskWrite': 1.011,
					'BwUp': 5000,
					'BwDown': 5000,
					'Power': 'PMRaspberryPi4B'
				},
			'RPi4B8G':
				{
					'IPS': 4029,
					'RAMSize': 8192,
					'RAMRead': 360.0,
					'RAMWrite': 305.0,
					'DiskSize': 32212,
					'DiskRead': 10.38,
					'DiskWrite': 0.619,
					'BwUp': 5000,
					'BwDown': 5000,
					'Power': 'PMRaspberryPi4B8G'
				}
 		}

	def generateHosts(self):
		hosts = []
		types = ['RPi4B'] * 8 + ['RPi4B8G'] * 8
		for i in range(self.num_hosts):
			typeID = types[i]
			IPS = self.types[typeID]['IPS']
			Ram = RAM(self.types[typeID]['RAMSize'], self.types[typeID]['RAMRead']*5, self.types[typeID]['RAMWrite']*5)
			Disk_ = Disk(self.types[typeID]['DiskSize'], self.types[typeID]['DiskRead']*5, self.types[typeID]['DiskWrite']*10)
			Bw = Bandwidth(self.types[typeID]['BwUp'], self.types[typeID]['BwDown'])
			Power = eval(self.types[typeID]['Power']+'()')
			Latency = 0.003 if i < self.edge_hosts else 0.076
			hosts.append((IPS, Ram, Disk_, Bw, Latency, Power))
		return hosts