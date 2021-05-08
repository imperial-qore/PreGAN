import os
import torch
import numpy as np
from .constants import *
from .models import *

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_to_windows(data, model):
	data = torch.tensor(data).double()
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def form_test_dataset(data):
	anomaly_per_dim = data > np.percentile(data, PERCENTILES, axis=0)
	anomaly_any_dim = np.logical_or.reduce(anomaly_per_dim, axis=1)
	anomaly_which_dim = []
	for i in range(0, data.shape[1], 3):
		anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
	anomaly_which_dim = np.stack(anomaly_which_dim).transpose()
	return anomaly_per_dim + 0, anomaly_which_dim

def load_npyfile(folder, fname):
	path = os.path.join(folder, fname)
	if not os.path.exists(path):
		raise Exception('Data not found ' + path)
	return np.load(path)

def load_dataset(folder, model):
	time_data = load_npyfile(folder, data_filename)
	train_schedule_data = torch.tensor(load_npyfile(folder, schedule_filename)).double()
	train_time_data = convert_to_windows(time_data, model)
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

def save_model(folder, fname, model, optimizer, epoch, accuracy_list):
	path = os.path.join(folder, fname)
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_prototypes': model.prototype,
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, path)

def load_model(folder, fname, modelname):
	import recovery.PreNetSrc.src.models
	model_class = getattr(recovery.PreNetSrc.src.models, modelname)
	model = model_class().double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.prototypes = checkpoint['model_prototypes']
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, epoch, accuracy_list