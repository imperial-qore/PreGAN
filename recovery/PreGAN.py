import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *

class PreGANRecovery(Recovery):
    def __init__(self, hosts):
        super().__init__()
        self.model_name = f'Attention_{hosts}'

    def train_model(self):
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def run_model(self, time_series, original_decision):
        # Load model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1:
            self.train_model()
        # Freeze encoder
        for name, p in self.model.named_parameters():
            if 'encoder' in name: p.requires_grad = False
        time_data, schedule_data = self.env.stats.time_series, torch.tensor(self.env.scheduler.result_cache).double()
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        anomaly, prototype = self.model(time_data, schedule_data)
        # If no anomaly predicted, return original decision 
        for a in anomaly:
            prediction = torch.argmax(a).item() 
            if prediction == 1: break
        else:
            return original_decision
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        embedding = torch.stack(embedding)
        print(embedding)

