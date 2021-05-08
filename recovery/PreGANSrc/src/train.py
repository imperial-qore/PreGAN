from .constants import *
from .utils import *
import torch.nn as nn
from tqdm import tqdm

anomaly_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss(reduction = 'mean')

# Model Training
def triplet_loss(anchor, positive_class, model):
	global PROTO_UPDATE_FACTOR
	positive_loss = mse_loss(anchor, model.prototype[positive_class].detach().clone())
	negative_class_list = [0, 1, 2]
	negative_class_list.remove(positive_class)
	negative_loss = []
	for nc in negative_class_list:
		negative_loss.append(mse_loss(anchor, model.prototype[nc]))
	loss = positive_loss - torch.sum(torch.tensor(negative_loss))
	if positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
		factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
		model.prototype[positive_class] = factor * anchor + (1 - factor) * model.prototype[positive_class]
	return loss

def custom_loss(model, source, target_anomaly, target_class):
	global PROTO_UPDATE_FACTOR
	source_anomaly, source_prototype = source
	aloss, tloss = 0, torch.tensor(0, dtype=torch.double)
	for i, sa in enumerate(source_anomaly):
		aloss += anomaly_loss(sa,  torch.tensor([target_anomaly[i]], dtype=torch.long))
	for i, sp in enumerate(source_prototype):
		if target_anomaly[i] > 0:
			tloss += triplet_loss(sp, target_class[i], model)
	PROTO_UPDATE_FACTOR *= PROTO_FACTOR_DECAY
	return aloss, tloss

def backprop(epoch, model, train_time_data, train_schedule_data, anomaly_data, class_data, optimizer, training = True):
	global PROTO_UPDATE_FACTOR
	if 'Attention' in model.name:
		aloss_list, tloss_list = [], []
		for i in tqdm(range(train_time_data.shape[0]), leave=False, position=1):
			output = model(train_time_data[i], train_schedule_data[i])
			aloss, tloss = custom_loss(model, output, anomaly_data[i], class_data[i])
			aloss_list.append(aloss.item()); tloss_list.append(tloss.item())
			loss = aloss + tloss
			if training:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		tqdm.write(f'Epoch {epoch},\tLoss = {loss},\tALoss = {np.mean(aloss_list)},\tTLoss = {np.mean(tloss_list)}')
		factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
		return np.mean(aloss_list) + np.mean(tloss_list), factor
	return


# Accuracy 
def anomaly_accuracy(source_anomaly, target_anomaly):
	correct = 0
	for i, sa in enumerate(source_anomaly):
		res = torch.argmax(sa).item() 
		if res == target_anomaly[i]:
			correct += 1
	return correct/len(source_anomaly)

def class_accuracy(source_prototype, target_anomaly, target_class, model):
	correct, total = 0, 1e-4
	for i, sp in enumerate(source_prototype):
		if target_anomaly[i] > 0:
			total += 1
			positive_loss = mse_loss(sp, model.prototype[target_class[i]])
			negative_class_list = [0, 1, 2]
			negative_class_list.remove(target_class[i])
			negative_loss = []
			for nc in negative_class_list:
				negative_loss.append(mse_loss(sp, model.prototype[nc]))
			if positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
				correct += 1
	return correct / total

def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data):
	anomaly_correct, class_correct, class_total = 0, 0, 0
	for i, d in enumerate(train_time_data):
		output = model(train_time_data[i], train_schedule_data[i])
		source_anomaly, source_prototype = output
		anomaly_correct += anomaly_accuracy(source_anomaly, anomaly_data[i])
		if np.sum(anomaly_data[i]) > 0:
			class_total += 1
			class_correct += class_accuracy(source_prototype, anomaly_data[i], class_data[i], model)
	return anomaly_correct / len(train_time_data), class_correct / class_total

