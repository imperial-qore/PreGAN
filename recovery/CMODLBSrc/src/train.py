from .constants import *
from .utils import *
import torch.nn as nn
from tqdm import tqdm

mse_loss = nn.MSELoss(reduction = 'mean')

# Model Training
def backprop(epoch, model, train_time_data, optimizer):
	loss_list = []
	for i in tqdm(range(train_time_data.shape[0] - 1), leave=False, position=1):
		output = model(train_time_data[i])
		loss = mse_loss(output, train_time_data[i+1])
		loss_list.append(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(loss_list)}')
	return np.mean(loss_list)
