import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics
import os, glob
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from .constants import *

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs(plot_folder, exist_ok=True)

def smoother(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class Model_Plotter():
	def __init__(self, env, modelname):
		self.env = env
		self.model_name = modelname
		self.n_hosts = int(modelname.split('_')[-1])
		self.folder = os.path.join(plot_folder, env, 'model')
		self.prefix = self.folder + '/' + self.model_name
		self.epoch = 0
		os.makedirs(self.folder, exist_ok=True)
		for f in glob.glob(self.folder + '/*'): os.remove(f)
		self.tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
		self.colors = ['r', 'g', 'b']
		self.init_params()

	def init_params(self):
		self.source_anomaly_scores = []
		self.target_anomaly_scores = []
		self.correct_series = []
		self.protoypes = []
		self.correct_series_class = []

	def update_anomaly(self, source_anomaly, target_anomaly, correct):
		self.source_anomaly_scores.append(source_anomaly)
		self.target_anomaly_scores.append(target_anomaly.tolist())
		self.correct_series.append(correct)

	def update_class(self, protoypes, correct):
		self.protoypes.extend(protoypes)
		self.correct_series_class.append(correct)

	def plot(self, accuracy_list, epoch):
		self.epoch = epoch; self.prefix2 = self.prefix + '_' + str(self.epoch) + '_'
		self.loss_list = [i[0] for i in accuracy_list]
		self.factor_list = [i[1] for i in accuracy_list]
		self.anomaly_score_list = [i[2] for i in accuracy_list]
		self.class_score_list = [i[3] for i in accuracy_list]
		self.plot1('Loss', self.loss_list)
		self.plot1('Factor', self.factor_list)
		self.plot2('Anomaly Prediction Score', 'Class Prediction Score', self.anomaly_score_list, self.class_score_list)
		self.plot1('Correct Anomaly', self.correct_series, xlabel='Timestamp')
		self.plot1('Correct Class', self.correct_series_class, xlabel='Timestamp')
		self.source_anomaly_scores = np.array(self.source_anomaly_scores)
		self.target_anomaly_scores = np.array(self.target_anomaly_scores)
		self.plot_heatmap('Anomaly Scores', 'Source', 'Target', self.source_anomaly_scores, self.target_anomaly_scores)
		X = [i[0].tolist() for i in self.protoypes]; Y = np.array([i[1] for i in self.protoypes])
		x2d = self.tsne.fit_transform(X)
		self.plot_tsne('Prototypes', x2d, Y)
		self.init_params()

	def plot1(self, name1, data1, smooth = True, xlabel='Epoch'):
		if smooth: data1 = smoother(data1)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax.set_xlabel(xlabel)
		fig.savefig(self.prefix2 + f'{name1}.pdf')
		plt.close()

	def plot2(self, name1, name2, data1, data2, smooth = True, xlabel='Epoch'):
		if smooth: data1, data2 = smoother(data1), smoother(data2)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax2 = ax.twinx()
		ax2.plot(data2, '--', linewidth=0.3, alpha=0.5)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf')
		plt.close()

	def plot_heatmap(self, title, name1, name2, data1, data2):
		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.set_title(title)
		ax1.set_ylabel(name1)
		ax1.set_xlabel('Timestamp')
		yticks = np.linspace(0, self.n_hosts, 10, dtype=np.int)
		h1 = sns.heatmap(data1.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax1)
		h2 = sns.heatmap(data2.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax2)
		ax1.set_yticks(yticks); ax2.set_yticks(yticks)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel(name2)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf')
		plt.close()

	def plot_tsne(self, name1, data1, labels1):
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		target_ids = range(3)
		for i, c, label in zip(target_ids, self.colors, labels1):
			ax.scatter(data1[labels1 == i, 0], data1[labels1 == i, 1], c=c, alpha=0.5, label=i)
		ax.legend()
		fig.savefig(self.prefix2 + f'tsne_{name1}.pdf')
		plt.close()

class GAN_Plotter():
	def __init__(self, env, gname, dname, training = True):
		self.env = env
		self.gname, self.dname = gname, dname
		self.n_hosts = int(gname.split('_')[-1])
		self.folder = os.path.join(plot_folder, env, 'gan' if training else 'test')
		self.prefix = self.folder + '/' + self.gname + '_' + self.dname
		self.epoch = 0
		os.makedirs(self.folder, exist_ok=True)
		for f in glob.glob(self.folder + '/*'): os.remove(f)
		self.init_params()

	def init_params(self):
		self.anomaly_detected = []
		self.class_detected = []
		self.hosts_migrated = []
		self.migrating = []
		self.new_score_better = []

	def update_anomaly_detected(self, detected):
		self.anomaly_detected.append(detected)

	def update_class_detected(self, detected):
		print(detected)
		self.class_detected.append(detected)

	def new_better(self, new_better):
		self.new_score_better.append(new_better + 0)
		if not new_better: 
			self.hosts_migrated.append([0] * int(self.gname.split('_')[1]))
			self.migrating.append(0)

	def plot_test(self, hosts_from):
		self.migrating.append((np.sum(hosts_from) > 0) + 0)
		self.hosts_migrated.append(hosts_from)
		self.prefix2 = self.prefix + '_Test_' + str(self.epoch) + '_'
		self.epoch += 1
		self.plot1('New Score Better', self.new_score_better)
		self.plot_heatmap('Anomaly Scores', 'Detection', 'Class', np.array(self.anomaly_detected).reshape(1, -1), np.array(self.class_detected))
		self.plot_heatmapc('Migrations', 'AnyMigrated', 'Hosts_from', np.array(self.migrating).reshape(1, -1), np.array(self.hosts_migrated))

	def plot(self, accuracy_list, epoch, ns, os):
		self.prefix2 = self.prefix + '_' + str(self.epoch) + '_'
		self.epoch += 1
		self.gloss_list = [i[0] for i in accuracy_list]
		self.dloss_list = [i[1] for i in accuracy_list]
		self.new_score_better.append((ns <= os) + 0)
		self.plot2('GLoss', 'DLoss', self.gloss_list, self.dloss_list)
		self.plot1('New Score Better', self.new_score_better)
		self.plot_heatmap('Anomaly Scores', 'Detection', 'Class', np.array(self.anomaly_detected).reshape(1, -1), np.array(self.class_detected))

	def plot1(self, name1, data1, smooth = True, xlabel='Epoch'):
		if smooth: data1 = smoother(data1)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax.set_xlabel(xlabel)
		fig.savefig(self.prefix2 + f'{name1}.pdf')
		plt.close()

	def plot2(self, name1, name2, data1, data2, smooth = True, xlabel='Epoch'):
		if smooth: data1, data2 = smoother(data1), smoother(data2)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax2 = ax.twinx()
		ax2.plot(data2, '--', linewidth=0.3, alpha=0.5)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf')
		plt.close()

	def plot_heatmap(self, title, name1, name2, data1, data2):
		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.set_title(title)
		ax1.set_ylabel(name1)
		ax1.set_xlabel('Timestamp')
		yticks = np.linspace(0, self.n_hosts, 10, dtype=np.int)
		h1 = sns.heatmap(data1,cmap="YlGnBu", yticklabels=[0], linewidth=0.01, ax = ax1)
		dcmap = LinearSegmentedColormap.from_list('Custom', ['w', 'r', 'g', 'b'], 4)
		data2 = (data2 + 1).transpose()
		h2 = sns.heatmap(data2,cmap=dcmap, yticklabels=yticks, linewidth=0.01, ax = ax2, vmin=0, vmax=3)
		colorbar = h2.collections[0].colorbar; colorbar.set_ticklabels(['None', 'CPU', 'RAM', 'Disk'])
		ax1.set_yticks([0]); ax2.set_yticks(yticks)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel(name2)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf')
		plt.close()

	def plot_heatmapc(self, title, name1, name2, data1, data2):
		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.set_title(title)
		ax1.set_ylabel(name1)
		ax1.set_xlabel('Timestamp')
		yticks = np.linspace(0, self.n_hosts, 10, dtype=np.int)
		data2 = data2.transpose()
		h1 = sns.heatmap(data1,cmap="YlGnBu", yticklabels=[0], linewidth=0.01, ax = ax1)
		h2 = sns.heatmap(data2,cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax2)
		ax1.set_yticks([0]); ax2.set_yticks(yticks)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel(name2)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf')
		plt.close()
