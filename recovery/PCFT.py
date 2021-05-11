import numpy as np
from .Recovery import *

class PCFTRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.utilHistory = []

    def updateUtilHistory(self):
        hostUtils = []
        for host in self.env.hostlist:
            hostUtils.append(host.getCPU())
        self.utilHistory.append(hostUtils)

    def recover_decision(self, original_decision):
        self.updateUtilHistory()
        host_selection = self.env.scheduler.LRSelection(self.utilHistory)
        if host_selection == []:
            return original_decision
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.env.scheduler.LeastFullPlacement(container_selection)
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
