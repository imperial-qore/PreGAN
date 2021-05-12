import numpy as np
from .Recovery import *

class ECLBRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.utilHistory = []
        self.maxExecTimeEstimate = 0
        self.lr_bw = 10

    def updateUtilHistory(self):
        hostUtils = []
        for host in self.env.hostlist:
            hostUtils.append(host.getCPU())
        self.utilHistory.append(hostUtils)
        execTimes = []
        for c in self.env.containerlist:
            if c: 
                execTimes.append(c.totalExecTime)
        self.maxExecTimeEstimate = max(execTimes)

    def predict_utilizations(self):
        current_util = np.array(self.utilHistory[-1])
        if len(self.utilHistory) < 2:
            return current_util        
        prev_util = np.array(self.utilHistory[-2])
        pred_util = 2 * current_util - prev_util
        return pred_util

    def select_hosts(self):
        pred_util = self.predict_utilizations()
        selected = []
        for i, p in enumerate(pred_util.tolist()):
            if p >= 100:
                selected.append(i)
        return selected

    def bayesian_target_selection(self, container_list):
        target_list = []
        for cid in container_list:
            estimate_times = []
            for host in self.env.hostlist:
                # calculate migration time
                ramsize = self.env.containerlist[cid].getContainerSize()
                bw = min(host.bwCap.downlink, self.env.containerlist[cid].getHost().bwCap.uplink)
                migration_time = ramsize / (bw + 1e-4)
                # calculate execution time
                exec_time = self.maxExecTimeEstimate - self.env.containerlist[cid].totalExecTime
                estimate_times.append(migration_time + exec_time)
            target_list.append((cid, np.argmin(estimate_times)))
        return target_list

    def recover_decision(self, original_decision):
        self.updateUtilHistory()
        host_selection = self.select_hosts()
        if host_selection == []:
            return original_decision
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.bayesian_target_selection(container_selection)
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
