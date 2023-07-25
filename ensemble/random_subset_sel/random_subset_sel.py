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

seed=42
random_state = seed_everything(seed=seed)

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
		return X, y
	X = SelectKBest(mutual_info_regression, k=k_feat).fit_transform(X,y)
	experiment_props.update({'filter':'True','k_feat':k_feat})
	return X, y


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
	if k_feat is not None:
		X, y = filter(X,y,k_feat)
	experiment_props['feats_selected']=X.shape[1]
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

def reference_matrix(model,X_train,X_test,y_train):
	m = X_train.shape[0]
	n = X_test.shape[0]
	ref = np.zeros((m,n))
	model.eval()
	with torch.no_grad():
		for i in range(m):
			for j in range(n):
				predicted = model(torch.unsqueeze(X_train[i],0),torch.unsqueeze(X_test[j],0),torch.unsqueeze(y_train[i],0))
				true = y_train[j]
				ref[i][j] = predicted.item()

	return ref

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


def plot_scatter(scatter_axes, fold, y_true, y_pred, pearson_r, mse):
	r, c = fold//plot_ncols, fold%plot_ncols
	scatter_axes[r,c].scatter(y_true, y_pred, color='blue', alpha=0.5)
	scatter_axes[r,c].plot(y_true, y_true, color='black', linewidth=1)
	scatter_axes[r,c].set_xlabel('Measured')
	scatter_axes[r,c].set_ylabel('Predicted')
	annotation = f"r = {pearson_r:.3f}\nMSE = {mse:.3f}"
	scatter_axes[r,c].text(0.02, 0.95, annotation, transform=scatter_axes[r,c].transAxes, fontsize=10, verticalalignment='top')
	scatter_axes[r,c].set_title(f'Fold {fold+1}')
	scatter_axes[r,c].tick_params(axis='both')
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

	num_epochs = 20
	batch_size = 16
	k = 3
	seed = 42

	pearson_feats =[]
	mse_feats = []

	num_trials = 3


	experiment_props = {'Model':'obscured', 'Trait': args.trait, 'epochs':num_epochs, 'batch_size':batch_size, 'folds':k}

	X, y = load_from_file(filename=filename,trait=args.trait)
	X, y = clean(X,y,threshhold=int(0.05*X.shape[1]),remove_redundance=True,window_len=8,k_feat=None) 
	print('After clean X.shape: ',X.shape)


	random_state = seed_everything(seed)
	splits=KFold(n_splits=k,shuffle=True,random_state=random_state)
	folds = splits.split(np.arange(X.shape[0]))
	pearson_folds = []
	mse_folds = []

	plot_nrows = math.ceil(k/2)
	plot_ncols = 2

	model_folds = []

	# fracs = [0.01,0.05,0.1,0.25,0.5,0.75,0.8,0.9,1]
	fracs = [0.01]+[i / 100 for i in range(5, 101, 5)]
	

	pearson_fracs = np.zeros(len(fracs))
	mse_fracs = np.zeros(len(fracs))


	with open(f'results/random_subset_sel/{args.dataset}/{args.trait}/fold_split_info.txt','w') as f:
		pass

	for fold, (train_idx,val_idx) in enumerate(folds):

		with open(f'results/random_subset_sel/{args.dataset}/{args.trait}/fold_split_info.txt','a') as f:
			f.write(f'Fold {fold}: train_idx={train_idx}\n')
			f.write(f'Fold {fold}: val_idx={val_idx}\n')

		random_state = seed_everything(seed)

		print('Fold {}, k_feat {}'.format(fold + 1, X.shape[1]))

		history = {'train_loss': [], 'test_loss': [], 'train_pearson': [], 'test_pearson': []}

		X_train, X_test = np.copy(X[train_idx]), np.copy(X[val_idx])
		y_train, y_test = np.copy(y[train_idx]), np.copy(y[val_idx])
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
			train_loss = train_loss / len(train_loader.dataset)
			

			if epoch % 10 == 9:
				test_loss, test_pearson =valid_epoch(model,device,test_loader,criterion)
				test_loss = test_loss / len(test_loader.dataset)
				print("Epoch:{}/{} AVG Training Loss:{:.3f} Training Pearson r:{:.3f} AVG Test Loss:{:.3f} Test Pearson r:{:.3f}".format(epoch + 1,num_epochs,train_loss,train_pearson,test_loss,test_pearson))


		# model_folds.append(copy.deepcopy(model))
		ref = reference_matrix(model,marker_train_matrix,marker_test_matrix,traits_train)

		######## plot the reference matrix ########
		fig, axis = plt.subplots(figsize=(0.1*marker_test_matrix.shape[0],0.1*marker_train_matrix.shape[0]))
		cmap = mpl.colormaps['YlOrBr']
		im = axis.imshow(np.abs(ref-y_test),cmap=cmap,vmin=np.min(np.abs(ref-y_test)),vmax=np.max(np.abs(ref-y_test)))
		axis.set_xticks(np.arange(marker_test_matrix.shape[0]))
		axis.set_yticks(np.arange(marker_train_matrix.shape[0]))
		axis.set_xticklabels(np.arange(marker_test_matrix.shape[0]))
		axis.set_yticklabels(np.arange(marker_train_matrix.shape[0]))
		fig.colorbar(im,ax=axis)
		axis.set_title(f'Fold {fold+1} reference matrix')
		fig.savefig(f'results/random_subset_sel/{args.dataset}/{args.trait}/fold_{fold+1}_ref_matrix.pdf')
		plt.close(fig)

		

		temp1 = []
		temp2 = []

		for frac_idx, frac in enumerate(fracs):
			pearson_trials = []
			mse_trials = []
			for trial in range(num_trials):
				subset_mask = random.sample(range(marker_train_matrix.shape[0]),int(frac*marker_train_matrix.shape[0]))
				y_true = []
				y_pred = []
				for testsample in range(marker_test_matrix.shape[0]):
					sum_pred = 0
					for reference in subset_mask:
						sum_pred += ref[reference][testsample]
					sum_pred /= len(subset_mask)
					y_pred.append(sum_pred)
					y_true.append(y_test[testsample])
				y_true = np.array(y_true)
				y_pred = np.array(y_pred)
				pearson_r = np.corrcoef(y_true, y_pred)[0,1]
				mse = np.mean((y_true-y_pred)**2)
				pearson_trials.append(pearson_r)
				mse_trials.append(mse)

			trial_avg_pearson = np.mean(pearson_trials)
			trial_avg_mse = np.mean(mse_trials)

			print(f'Fold {fold+1} fraction={frac} pearson_r={trial_avg_pearson} mse={trial_avg_mse}')
			# plot_scatter(scatter_axes[frac_idx], fold, y_true, y_pred,pearson_r,mse)
			temp1.append(trial_avg_pearson)
			temp2.append(trial_avg_mse)

		pearson_fracs += np.array(temp1)
		mse_fracs += np.array(temp2)

	pearson_fracs /= k
	mse_fracs /= k


	with open(f'results/random_subset_sel/{args.dataset}/{args.trait}/summary.csv','w') as f: # to erase out the previous contents of the file
		pass

	with open(f'results/random_subset_sel/{args.dataset}/{args.trait}/summary.csv','a') as f:
		f.write(f'Frac_sel,Mean_PCC,Mean_MSE\n')
		for i in range(len(fracs)):
			f.write(f'{fracs[i]},{pearson_fracs[i]},{mse_fracs[i]}\n')

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
	plt.subplots_adjust(wspace=0.2,hspace=0.3)

	axes[0].plot(fracs,pearson_fracs,linewidth=1,color='red')
	axes[0].scatter(fracs,pearson_fracs, color='red', marker='s')
	axes[0].set_xlabel('Fraction of train samples')
	axes[0].set_ylabel('Pearson_r')
	axes[0].set_title('Pearson_r vs Fraction of train samples')

	axes[1].plot(fracs,mse_fracs,linewidth=1,color='blue')
	axes[1].scatter(fracs,mse_fracs, color='blue', marker='s')
	axes[1].set_xlabel('Fraction of train samples')
	axes[1].set_ylabel('MSE')
	axes[1].set_title('MSE vs Fraction of train samples')

	plt.savefig(f'results/random_subset_sel/{args.dataset}/{args.trait}/Comparison_across_different_samples.pdf')
