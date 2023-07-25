import random
import pickle
import os
import math
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

# data.py

def load_from_file(filename, traits=['flr','mat','ht','yld','pc','oc','gluc'], trait_idx=None, trait=None):
	df = pd.read_csv(filename)
	df_x = df.iloc[:,2:len(df.columns)-len(traits)]
	df_y = df.iloc[:,len(df.columns)-len(traits):]

	if trait is None:
		trait = traits[trait_idx]

	df_comb = pd.concat([df_x,df_y[trait]],axis=1)
	df_comb = df_comb.dropna()
	X = np.array(df_comb.iloc[:,0:len(df_comb.columns)-1])
	y = np.array(df_comb.iloc[:,len(df_comb.columns)-1])

	return X, y

def remove_missing_rows(X, y, threshhold):
	# remove those rows where count(-2) >= threshhold
	experiment_props['missing_vals_threshold']=threshhold
	mask = []
	for i in range(X.shape[0]):
		if np.count_nonzero(X[i]==-2) < threshhold:
			mask.append(i)

	X = X[mask]
	y = y[mask]

	return X, y

def make_col_continuous(X,X_cont,col):
	D = np.count_nonzero(X[:,col]==1)
	H = np.count_nonzero(X[:,col]==0)
	N = X.shape[0]
	# print(col,':',D,H,N)
	p = (2 * D + H) / (2 * N)
	q = 1 - p
	replace_with = [2*p*q, p**2, q**2]
	# print(replace_with)
	for i in range(X.shape[0]):
		X_cont[i][col] = replace_with[X[i][col]]
	return X_cont

def make_mat_continuous(X):
	X_cont = np.empty_like(X, dtype=float)
	for j in range(X.shape[1]):
		X_cont = make_col_continuous(X,X_cont,j)

	return X_cont

def prep_data(X,y):
	sim_data = []
	for i in range(len(X)):
		for j in range(i,len(X)):
			add_data = []
			add_data.append([X[i],X[j]])
			# randomly pickup a number from {0,1}
			if np.random.randint(2) == 0:
				add_data.append(y[i])
				add_data.append(y[j])			
								
			else:
				add_data.append(y[j])
				add_data.append(y[i])
			sim_data.append(add_data)
	# shuffle the data
	random.shuffle(sim_data)
	return sim_data

def calculate_redundancy_mask(X,threshold_fraction,window_len,verbose=False):
	threshold = X.shape[0]*threshold_fraction

	cnt = 0
	maxdiff = 0
	dist = {}

	flag = np.ones(X.shape[1],dtype=np.int8)
	for i in (range(0,X.shape[1])):
		if flag[i]==0:
			continue
		for j in range(i+1,i+1+window_len):
			if j>=X.shape[1]:
				break
			if flag[j]==0:
				continue
			x1 = X[:,i]
			x2 = X[:,j]

			x3 = np.array([1 if (x1[k]==x2[k]) else 0 for k in range(X.shape[0])])
			x4 = np.array([1 if (x1[k]==-1*x2[k]) else 0 for k in range(X.shape[0])])
			if np.sum(x3)>=threshold or np.sum(x4)>=threshold:
				flag[j]=0
				maxdiff = max(maxdiff,j-i)
				range_x = ((j-i)//5)*5
				range_y = range_x+5
				if (range_x,range_y) in dist:
					dist[(range_x,range_y)] += 1
				else:
					dist[(range_x,range_y)] = 1
				cnt += 1
	if verbose:
		print('\n(1) cnt =',cnt)   
		print('(2) maxdiff =',maxdiff) 
		print('(3) Distribution of correlated columns:')
		for range_x in range(0,(maxdiff//5)+1):
			print(f'[{range_x*5},{range_x*5+5}]: {dist.get((range_x*5,range_x*5+5),0)}')

		print(f'Number of cols retained = {np.sum(flag)}')

	# indices of retained columns
	indices = np.nonzero(flag)[0]
	return indices

def filter(X,y,k_feat):
	if X.shape[1]<=k_feat:
		experiment_props.update({'filter':'False','k_feat':'-'})
		return np.ones(X.shape[1],dtype=bool)
	experiment_props.update({'filter':'True','k_feat':k_feat})
	filter_mask = SelectKBest(mutual_info_regression, k=k_feat).fit(X,y).get_support()
	return filter_mask
	

def clean(X,y,threshhold,remove_redundance=True,redundancy_mask=None,threshold_fraction=1,window_len=1,k_feat=None):
	X, y = remove_missing_rows(X, y, threshhold=threshhold) # remove columns with missing marker count > threshhold
	X = SimpleImputer(missing_values=-2, strategy='most_frequent').fit_transform(X) # impute remaining missing values
	experiment_props['imputation_strategy']='most_frequent'
	if remove_redundance:
		if redundancy_mask is None:
			experiment_props.update({'remove_redundace': 'True','threshold_fraction':threshold_fraction,'window_len':window_len,'mask_file':'-'})
			redundancy_mask = calculate_redundancy_mask(X,threshold_fraction,window_len)
		else:
			experiment_props.update({'remove_redundace': 'True','threshold_fraction':'-','window_len':'-','mask_file':redundancy_mask})
		X = X[:,redundancy_mask] # change
	else:
		experiment_props.update({'remove_redundace': 'False','threshold_fraction':'-','window_len':'-','mask_file':'-'})
	# X = make_mat_continuous(X) # make catergorical data continuous with hardy wienberg
	experiment_props.update({'retained_cols': X.shape[1]})
	# if k_feat is not None:
	# 	X, y = filter(X,y,k_feat)
	# experiment_props['feats_selected']=X.shape[1]
	return X.astype(np.int32), y.astype(np.float32)

class MarkerDataset(Dataset):
	def __init__(self, X, y, transform=None, target_transform=None):
		self.geno = X
		self.pheno = y
		self.num_features = X.shape[1]
		self.data = prep_data(X,y)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		seqs_a = self.data[idx][0][0]
		seqs_b = self.data[idx][0][1]
		pheno_a = self.data[idx][1]
		pheno_b = self.data[idx][2]
		if self.transform:
			seqs_a = self.transform(seqs_a)
			seqs_b = self.transform(seqs_b)
		if self.target_transform:
			pheno_a = self.target_transform(pheno_a)
			pheno_b = self.target_transform(pheno_b)
		return (seqs_a, seqs_b, pheno_a), pheno_b
	
# model.py

def get_conv_output_size(w, padding, kernel_size, stride):
	return int(((w - kernel_size + (2 * padding)) / stride) + 1)


class MatchingModel(nn.Module):
	def __init__(self, sample_length, num_hidden_estimator, d=1):
		super(MatchingModel, self).__init__()

		self.sample_length = sample_length
		conv1_output_size = get_conv_output_size(sample_length, 9, 18, 1)

		self.extractor = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1, padding=9),
			nn.ReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.Dropout(p=0.2),
			nn.Linear(int(conv1_output_size / 4) * 8, 32),
			nn.ReLU(),
			nn.Dropout(p=0.05),
			nn.Linear(32, d)
		)

		self.estimator = nn.Sequential(
			nn.Linear(d+1, num_hidden_estimator, bias=False),
			nn.BatchNorm1d(num_hidden_estimator),
			nn.ReLU(),
			nn.Linear(num_hidden_estimator, 1)
		)

	def forward(self, seqs_a, seqs_b, pheno_a):
		matches = (seqs_a == seqs_b).to(torch.float32).clone()
		feats = self.extractor(torch.unsqueeze(matches, dim=1))
		# print(feats.shape)
		# print(feats.dtype)
		# print(pheno_a.dtype)
		estimator_input = torch.cat((pheno_a[:, None], feats), dim=1)
		# print(estimator_input.dtype)
		y = self.estimator(estimator_input)
		return torch.squeeze(y)

def train_epoch(model,device,dataloader,loss_fn,optimizer):
	train_loss = 0.0
	y_true = []
	y_pred = []
	model.train()
	for (seqs_a, seqs_b, pheno_a), pheno_b in dataloader:

		seqs_a,seqs_b,pheno_a,pheno_b = seqs_a.to(device),seqs_b.to(device),pheno_a.to(device),pheno_b.to(device)
		optimizer.zero_grad()
		output = model(seqs_a,seqs_b,pheno_a)
		loss = loss_fn(output,pheno_b)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * pheno_a.size(0)
		y_true.extend(pheno_b.tolist())
		y_pred.extend(output.tolist())
	pearson = np.corrcoef(y_pred,y_true)[0,1]
	
	return train_loss, pearson # return cumulative trainin loss across all training data and pearson correlation

def valid_epoch(model,device,dataloader,loss_fn):
	valid_loss = 0.0
	y_true = []
	y_pred = []
	model.eval()
	with torch.no_grad():
		for (seqs_a, seqs_b, pheno_a), pheno_b in dataloader:
			seqs_a,seqs_b,pheno_a,pheno_b = seqs_a.to(device),seqs_b.to(device),pheno_a.to(device),pheno_b.to(device)
			output = model(seqs_a,seqs_b,pheno_a)
			loss = loss_fn(output,pheno_b)
			valid_loss += loss.item() * pheno_a.size(0)
			y_true.extend(pheno_b.tolist())
			y_pred.extend(output.tolist())
	pearson = np.corrcoef(y_pred,y_true)[0,1]

	return valid_loss, pearson # return cumulative validation loss across all validation data and pearson correlation

# inference.py

def predict(geno_a, marker_train_matrix, traits_train, model, device):
	avg_numerator = 0.0
	avg_denominator = marker_train_matrix.shape[0]
	model.eval()
	with torch.no_grad():
		for idx, geno_b in enumerate(marker_train_matrix):
			output = model(torch.unsqueeze(geno_a,0), torch.unsqueeze(geno_b,0), torch.unsqueeze(traits_train[idx],0))
			avg_numerator += output.item()
	return (avg_numerator / avg_denominator)

def predict_all(marker_test_matrix, marker_train_matrix, traits_train, model, device):
	predictions = []
	for geno_a in marker_test_matrix:
		predictions.append(predict(geno_a, marker_train_matrix, traits_train, model, device))
	return predictions

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

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	criterion = nn.MSELoss()

	num_epochs = 20
	batch_size = 16
	k = 3
	seed = 42

	pearson_feats =[]
	mse_feats = []

	experiment_props = {'Model':'obscured', 'Trait': args.trait, 'epochs':num_epochs, 'batch_size':batch_size, 'folds':k}

	X, y = load_from_file(filename,trait=args.trait)
	X, y = clean(X,y,threshhold=int(0.05*X.shape[1]),remove_redundance=True,window_len=8,k_feat=None) 
	print('After clean(removal of rows with high missing values + redundancy removal) X.shape: ',X.shape)

	k_feats = [2**i for i in range(3,int(math.log2(X.shape[1]))+1)]

	with open(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/fold_split_info.txt','w') as f:
		pass 

	with open(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/obscured.csv','w') as f:
		f.write('Cols retained,pearson_r,mse\n')

	for k_feat in k_feats:

		random_state = seed_everything(seed)
		splits=KFold(n_splits=k,shuffle=True,random_state=random_state)
		pearson_folds = []
		mse_folds = []

		plot_nrows = math.ceil(k/2)
		plot_ncols = 2

		fig, axes = plt.subplots(nrows=plot_nrows, ncols=plot_ncols, figsize=(plot_nrows*5,plot_nrows*6))
		plt.subplots_adjust(wspace=0.2,hspace=0.3)


		for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(X.shape[0]))):

			with open(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/fold_split_info.txt','a') as f:
				f.write(f'k_feat={k_feat},fold={fold+1}\n')
				f.write(f'train_idx={train_idx}\n')
				f.write(f'val_idx={val_idx}\n')

			print('Fold {}, k_feat {}'.format(fold + 1, k_feat))

			history = {'train_loss': [], 'test_loss': [], 'train_pearson': [], 'test_pearson': []}

			X_train, X_test = np.copy(X[train_idx]), np.copy(X[val_idx])
			y_train, y_test = np.copy(y[train_idx]), np.copy(y[val_idx])

			# filter based feature selection
			filter_mask = filter(X_train,y_train,k_feat)
			X_train = X_train[:,filter_mask]
			X_test = X_test[:,filter_mask]

			y_meu, y_std = np.mean(y_train), np.std(y_train)
			
			# normalize y_train and y_test
			y_train = (y_train - y_meu) / y_std
			y_test = (y_test - y_meu) / y_std

			train_dataset = MarkerDataset(X_train, y_train, transform=torch.from_numpy, target_transform=None)
			test_dataset = MarkerDataset(X_test, y_test, transform=torch.from_numpy, target_transform=None)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

			print('X_train.shape: ', X_train.shape) # change
			print('train_loader.dataset len: ', len(train_loader.dataset)) # change

			marker_train_matrix = torch.from_numpy(np.copy(X_train)).to(device)
			traits_train = torch.from_numpy(np.copy(y_train)).to(device)
			marker_test_matrix = torch.from_numpy(np.copy(X_test)).to(device)
			traits_test = torch.from_numpy(np.copy(y_test)).to(device)
			
			model = MatchingModel(sample_length=train_dataset.num_features, num_hidden_estimator=32, d=1)
			model.to(device)
			lr=0.001
			optimizer = optim.Adam(model.parameters(), lr=lr)
			experiment_props.update({'optimizer':'Adam','loss':'MSE'})

			for epoch in range(num_epochs):
				train_loss, train_pearson =train_epoch(model,device,train_loader,criterion,optimizer)
				test_loss, test_pearson =valid_epoch(model,device,test_loader,criterion)

				train_loss = train_loss / len(train_loader.dataset)
				test_loss = test_loss / len(test_loader.dataset)

				print("Epoch:{}/{} AVG Training Loss:{:.3f} Training Pearson r:{:.3f} AVG Test Loss:{:.3f} Test Pearson r:{:.3f}".format(epoch + 1,num_epochs,train_loss,train_pearson,test_loss,test_pearson))

				

				if epoch % num_epochs == num_epochs-1:
					y_true = np.array(traits_test.cpu())
					y_pred = np.array(predict_all(marker_test_matrix, marker_train_matrix, traits_train, model, device))
					pearson_r = np.corrcoef(y_true, y_pred)[0,1]
					mse = np.mean((y_true-y_pred)**2)
					plot_scatter(fold, k, y_true, y_pred,pearson_r,mse)
					print('Estimating traits of samples from test set using training samples as reference:')
					print("Pearson r: {:.3f}".format(pearson_r))
					print("MSE: {:.3f}".format(mse))			
					pearson_folds.append(pearson_r)
					mse_folds.append(mse)

				history['train_loss'].append(train_loss)
				history['test_loss'].append(test_loss)
				history['train_pearson'].append(train_pearson)
				history['test_pearson'].append(test_pearson)
		
		print(f'Average pearson_r accross all folds = {np.mean(np.array(pearson_folds))}')
		print(f'Average mse accross all folds = {np.mean(np.array(mse_folds))}')
		pearson_feats.append(np.mean(np.array(pearson_folds)))
		mse_feats.append(np.mean(np.array(mse_folds)))
		experiment_props.update({'Avg_PCC':np.mean(np.array(pearson_folds)),'Avg_MSE':np.mean(np.array(mse_folds))})
		plot_title = f'Features retained = {X_train.shape[1]}'
		fig.suptitle(str(experiment_props),wrap=True)
		plt.savefig(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/cols_{X_train.shape[1]}.pdf')
		plt.close()
		with open(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/obscured.csv','a') as f:
			f.write(f'{X_train.shape[1]},{np.mean(np.array(pearson_folds))},{np.mean(np.array(mse_folds))}\n')
	
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
	plt.subplots_adjust(wspace=0.2,hspace=0.3)

	axes[0].plot(k_feats,pearson_feats,linewidth=1,color='red')
	axes[0].scatter(k_feats,pearson_feats, color='red', marker='s')
	axes[0].set_xlabel('#Features')
	axes[0].set_ylabel('Pearson_r')
	axes[0].set_xscale('log')
	axes[0].set_title('Pearson_r vs #Features')

	axes[1].plot(k_feats,mse_feats,linewidth=1,color='blue')
	axes[1].scatter(k_feats,mse_feats, color='blue', marker='s')
	axes[1].set_xlabel('#Features')
	axes[1].set_ylabel('MSE')
	axes[1].set_xscale('log')
	axes[1].set_title('MSE vs #Features')

	plt.savefig(f'results/feature_sel/{args.dataset}/{args.trait}/obscured/inflection_plot.pdf')