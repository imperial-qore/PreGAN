import numpy as np
from .Recovery import *
import math
from utils.MathUtils import *
from utils.MathConstants import *

class DFTMRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.utilHistory = []
        self.lr_bw = 10

    def updateUtilHistory(self):
        hostUtils = []
        for host in self.env.hostlist:
            hostUtils.append(host.getCPU())
        self.utilHistory.append(hostUtils)

    def predict_utilizations(self):
        if (len(self.utilHistory) < self.lr_bw):
            return self.env.scheduler.ThresholdHostSelection()
        selectedHostIDs = []; x = list(range(self.lr_bw))
        for i,host in enumerate(self.env.hostlist):
            hostL = [self.utilHistory[j][i] for j in range(len(self.utilHistory))]
            _, estimates = loess(x, hostL[-self.lr_bw:], poly_degree=1, alpha=0.6)
            weights = estimates['b'].values[-1]
            predictedCPU = weights[0] + weights[1] * (self.lr_bw + 1)
            if 1.2 * predictedCPU >= 100:
                selectedHostIDs.append((predictedCPU, i))
        # Take maximum 4% hosts based on cpu utilization
        selectedHostIDs = sorted(selectedHostIDs, reverse = True)
        if len(selectedHostIDs) > 0.04 * self.hosts:
            selectedHostIDs = selectedHostIDs[:int(0.04 * self.hosts)]
        return [i[0] for i in selectedHostIDs]

    def recover_decision(self, original_decision):
        self.updateUtilHistory()
        host_selection = self.predict_utilizations()
        if host_selection == []:
            return original_decision
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.env.scheduler.FirstFitPlacement(container_selection)
        container_alloc = [-1] * len(self.env.hostlist)
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision)
        for cid, hid in target_selection:
            if container_alloc[cid] != hid:
                decision_dict[cid] = hid
        return list(decision_dict.items())

    def run_model(self, time_series, original_decision):
        return self.recover_decision(original_decision)
