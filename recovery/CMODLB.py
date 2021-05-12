import sys
sys.path.append('recovery/CMODLBSrc/')

import numpy as np
from sklearn.cluster import KMeans
from .Recovery import *
from .CMODLBSrc.src.constants import *
from .CMODLBSrc.src.utils import *
from .CMODLBSrc.src.train import *

class CMODLBRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.model_name = f'FCN_{hosts}'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.kmeans = KMeans(n_clusters=2, random_state=0)
        self.load_model()
        self.detections = (0, 0)

    def load_model(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model)
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
        folder = os.path.join(data_folder, self.env_name)
        train_time_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss = backprop(self.epoch, self.model, train_time_data, self.optimizer)
            self.accuracy_list.append(loss)
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def run_encoder(self):
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return self.model(time_data)

    def recover_decision(self, pred_values, original_decision):
        kmean_lists = pred_values.reshape(self.hosts, 3)
        kmeans = self.kmeans.fit(kmean_lists)
        self.detections = (self.detections[0], self.detections[1] + 1)
        # Single label found
        if len(np.unique(kmeans.labels_)) == 1:
            return original_decision
        # Clusters too close
        if abs(np.sum(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])) < (0.025 * self.hosts):
            return original_decision
        self.detections = (self.detections[0] + 1, self.detections[1])
        if np.sum(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 0:
            higher, lower = 0, 1
        else:
            higher, lower = 1, 0
        host_selection = []
        for host_embedding in range(kmean_lists.shape[1]):
            if kmeans.labels_[host_embedding] == higher:
                host_selection.append(host_embedding)
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
        print(f"{color.HEADER}Percent Detection: {int(100 * self.detections[0] / self.detections[1])}, \
            Hosts: {len(host_selection)}{color.ENDC}")
        return list(decision_dict.items())


    def run_model(self, time_series, original_decision):
        # Run encoder
        pred_values = self.run_encoder()
        return self.recover_decision(pred_values, original_decision)

