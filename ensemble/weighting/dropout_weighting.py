import random
import pickle
import os
import math
import copy
import argparse

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
from torchvision import transforms



def seed_everything(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	pd.set_option("mode.chained_assignment", None)  # Disable pandas warnings
	# pd.np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	random_state = check_random_state(seed)

	return random_state

# model.py

class ReferenceModel(nn.Module):
	def __init__(self, dropout_rate, ref_cnt):
		super(ReferenceModel, self).__init__()

		self.dropout_rate = dropout_rate
		self.ref_cnt = ref_cnt
		

		self.estimator = nn.Sequential(
			nn.Dropout(p=self.dropout_rate),
			nn.Linear(self.ref_cnt, 1)
		)

	def forward(self, refs):
		inputs = refs.to(torch.float32).clone()
		pred = self.estimator(inputs)
		return torch.squeeze(pred, dim=1)
	
class ReferenceDataset(Dataset):
	def __init__(self, X, y, transform=None, target_transform=None):
		self.refs = X
		self.vals = y
		self.ref_cnt = X.shape[1]
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.vals)

	def __getitem__(self, idx):
		feat = self.refs[idx]
		label = self.vals[idx]
		if self.transform:
			feat = self.transform(feat)
		if self.target_transform:
			label = self.target_transform(label)
		return feat, label
	
def train_epoch(model,device,dataloader,loss_fn,optimizer):
	train_loss = 0.0
	y_true = []
	y_pred = []
	model.train()
	for feats, labels in dataloader:

		feats, labels = feats.to(device),labels.to(device)
		optimizer.zero_grad()
		output = model(feats)
		loss = loss_fn(output,labels)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * labels.size(0)
		y_true.extend(labels.tolist())
		y_pred.extend(output.tolist())
	pearson = np.corrcoef(y_pred,y_true)[0,1]

	return train_loss, pearson

def valid_epoch(model,device,dataloader,loss_fn):
	valid_loss = 0.0
	y_true = []
	y_pred = []
	model.eval()
	with torch.no_grad():
		for feats, labels in dataloader:
			feats, labels = feats.to(device),labels.to(device)
			output = model(feats)
			loss = loss_fn(output,labels)
			valid_loss += loss.item() * labels.size(0)
			y_true.extend(labels.tolist())
			y_pred.extend(output.tolist())
	pearson = np.corrcoef(y_pred,y_true)[0,1]

	return valid_loss, pearson # return cumulative validation loss across all validation data and pearson correlation

def predict_all(model, train_test):
	y_pred = []
	model.eval()
	with torch.no_grad():
		for refs in train_test:
			refs = torch.from_numpy(refs).to(device)
			output = model(torch.unsqueeze(refs,0))
			y_pred.append(output.item())

	return y_pred

def plot_scatter(fold, splits, y_true, y_pred, pearson_r, mse):
	r, c = fold//plot_ncols, fold%plot_ncols
	axes[r,c].scatter(y_true, y_pred, color='blue', alpha=0.5)
	axes[r,c].plot(y_true, y_true, color='black', linewidth=1)
	axes[r,c].set_xlabel('Measured')
	axes[r,c].set_ylabel('Predicted')
	annotation = f"r = {pearson_r:.3f}\nMSE = {mse:.3f}"
	axes[r,c].text(0.02, 0.95, annotation, transform=axes[r,c].transAxes, fontsize=10, verticalalignment='top')
	axes[r,c].set_title(f'Fold {fold+1}')
	axes[r,c].tick_params(axis='both')
	# plt.show()
	# plt.savefig('fold_{}.svg'.format(fold+1))
	# plt.close()

# evaluate.py

if __name__ == '__main__':

	# downloading font for rendering matplotlib text
	fm.fontManager.addfont('./fontdir/LibreBaskerville-Regular.ttf')
	mpl.rc('font', family='Libre Baskerville')

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--trait', type=str)

	args = parser.parse_args()

	if args.dataset == 'canola':
		filename = 'canola_data.csv'
		traits = ['flr','mat','ht','yld','pc','oc','gluc']
		if args.trait not in traits:
			pass # handle invalid trait

	else:
		filename = 'lentil_ldp_lr01_data.csv'
		traits = ['DTF','VEG','DTM','DTS','REP']
		if args.trait not in traits:
			pass # handle invalid trait

	seed=42
	random_state = seed_everything(seed=seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	criterion = nn.MSELoss()

	with open(f'training_cache/{args.dataset}/{args.trait}/fold_info.pickle', 'rb') as f:
		fold_info = pickle.load(f)

	num_epochs = 20
	batch_size = 16

	# fracs = fracs = [0.01,0.05,0.1,0.25,0.5,0.75,0.8,0.9,1]
	fracs = [0.01]+[i / 100 for i in range(5, 101, 5)]
	dropout_rates = [1-fracs[i] for i in range(len(fracs))]
	k = len(fold_info)

	mse_fracs = []
	pearson_fracs = []

	for idx, dropout_rate in enumerate(dropout_rates):
		print(f'Dropout rate: {dropout_rate}')
		fold_cnt = 0
		pearson_sum = 0
		mse_sum = 0

		plot_nrows = math.ceil(k/2)
		plot_ncols = 2

		fig, axes = plt.subplots(nrows=plot_nrows, ncols=plot_ncols, figsize=(plot_nrows*5,plot_nrows*6))
		plt.subplots_adjust(wspace=0.2,hspace=0.3)

		for fold, fold_dict in enumerate(fold_info):
			
			fold_cnt += 1
			print(f'Fold {fold_dict["fold"]}')
			
			train_train = fold_dict['train_train'] # size: (n_train,n_train)
			train_test = fold_dict['train_test'] # size: (n_train,n_test)

			y_train = fold_dict['y_train'] # size: (n_train,)
			y_test = fold_dict['y_test'] # size: (n_test,)


			train_train = np.transpose(train_train) # size: (n_train,n_train)
			train_test = np.transpose(train_test) # size: (n_test,n_train)

			# declare dataset and dataloaders
			train_dataset = ReferenceDataset(train_train, y_train, transform=torch.from_numpy, target_transform=None)
			test_dataset = ReferenceDataset(train_test, y_test, transform=torch.from_numpy, target_transform=None)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

			# declare model
			model = ReferenceModel(dropout_rate, train_train.shape[1]).to(device)

			# train model
			optimizer = optim.Adam(model.parameters(), lr=0.001)

			for epoch in range(num_epochs):
				train_loss, train_pearson = train_epoch(model, device, train_loader, criterion, optimizer)
				valid_loss, valid_pearson = valid_epoch(model, device, test_loader, criterion)

				train_loss = train_loss / len(train_loader.dataset)
				valid_loss = valid_loss / len(test_loader.dataset)

				print(f'Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Train Pearson: {train_pearson:.4f} | Valid loss: {valid_loss:.4f} | Valid Pearson: {valid_pearson:.4f}')


			# evaluate y_true and y_pred
			y_pred = np.array(predict_all(model, train_test))
			y_true = y_test

			mse = np.mean((y_true - y_pred)**2)
			pearson_r = np.corrcoef(y_true, y_pred)[0,1]
			plot_scatter(fold, k, y_true, y_pred,pearson_r,mse)
			mse_sum += mse
			pearson_sum += pearson_r

			print(f'MSE: {mse}')
			print(f'Pearson r: {pearson_r}')
		
		print(f'Dropout rate: {dropout_rate}, Retained fraction: {fracs[idx]}: MSE: {mse_sum/fold_cnt}, Pearson r: {pearson_sum/fold_cnt}\n')
		mse_fracs.append(mse_sum/fold_cnt)
		pearson_fracs.append(pearson_sum/fold_cnt)

		plot_title = f'Dropout = {dropout_rate}, retained fraction = {fracs[idx]}'
		fig.suptitle(plot_title,wrap=True)
		plt.savefig(f'results/dropout_weighting/{args.dataset}/{args.trait}/retained_frac_{fracs[idx]}.pdf')
		plt.close()

	with open(f'results/dropout_weighting/{args.dataset}/{args.trait}/summary.csv','w') as f: # to erase out the previous contents of the file
		pass

	with open(f'results/dropout_weighting/{args.dataset}/{args.trait}/summary.csv','a') as f:
		f.write(f'Frac_sel,Mean_PCC,Mean_MSE\n')
		for i in range(len(fracs)):
			f.write(f'{fracs[i]},{pearson_fracs[i]},{mse_fracs[i]}\n')

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
	plt.subplots_adjust(wspace=0.2,hspace=0.3)

	axes[0].plot(fracs,pearson_fracs,linewidth=1,color='red')
	axes[0].scatter(fracs,pearson_fracs, color='red', marker='s')
	axes[0].set_xlabel('1-dropout_rate')
	axes[0].set_ylabel('Pearson_r')
	axes[0].set_title('Pearson_r vs (1-dropout_rate)')

	axes[1].plot(fracs,mse_fracs,linewidth=1,color='blue')
	axes[1].scatter(fracs,mse_fracs, color='blue', marker='s')
	axes[1].set_xlabel('1-dropout_rate')
	axes[1].set_ylabel('MSE')
	axes[1].set_title('MSE vs (1-dropout_rate)')

	plt.savefig(f'results/dropout_weighting/{args.dataset}/{args.trait}/Comparison_across_different_samples.pdf')