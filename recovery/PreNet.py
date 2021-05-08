import sys
sys.path.append('recovery/PreNetSrc/')

import numpy as np
from .Recovery import *
from .PreNetSrc.src.constants import *
from .PreNetSrc.src.utils import *
from .PreNetSrc.src.train import *

class PreNetRecovery(Recovery):
    def __init__(self, hosts):
        super().__init__()
        self.model_name = f'Attention_{hosts}'
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)

    def train_model(self):
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def run_model(self, time_series, decision):
        if self.epoch == -1:
            self.train_model()
        pass

    def anomaly_prediction(self, ):
        pass
