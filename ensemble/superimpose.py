import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse

def extract(filename):
	df = pd.read_csv(filename)
	feat = np.array(df['Frac_sel'])
	pearson = np.array(df['Mean_PCC'])
	mse = np.array(df['Mean_MSE'])
	return feat, pearson, mse


if __name__=='__main__':


	# downloading font for rendering matplotlib text
	fm.fontManager.addfont('./fontdir/LibreBaskerville-Regular.ttf')
	mpl.rc('font', family='Libre Baskerville')

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--trait', type=str)

	args = parser.parse_args()

	dir = f'{args.dataset}/{args.trait}/'


	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
	plt.subplots_adjust(wspace=0.2,hspace=0.3)

	feat1, pearson1, mse1 = extract('diff_matrix_sel/results/diff_matrix_sel/'+dir+'summary.csv')
	feat2, pearson2, mse2 = extract('random_subset_sel/results/random_subset_sel/'+dir+'summary.csv')
	# feat3, pearson3, mse3 = extract('weighting/results/dropout_weighting/'+dir+'summary.csv')


	feat1 = feat1[1:]
	pearson1 = pearson1[1:]
	mse1 = mse1[1:]

	feat2 = feat2[1:]
	pearson2 = pearson2[1:]
	mse2 = mse2[1:]

	# feat3 = feat3[1:]
	# pearson3 = pearson3[1:]
	# mse3 = mse3[1:]

	axes[0].plot(feat1,pearson1,color='red',label='Matrix selection')
	axes[0].scatter(feat1,pearson1, color='red', marker='o')

	axes[0].plot(feat2,pearson2,color='blue',label='Random selection')
	axes[0].scatter(feat2,pearson2, color='blue', marker='o')

	# axes[0].plot(feat3,pearson3,color='green',label='dropout')
	# axes[0].scatter(feat3,pearson3, color='green', marker='o')

	axes[0].set_xlabel('Fraction of samples selected')
	axes[0].set_ylabel('Pearson correlation coefficient')
	axes[0].set_title('Pearson correlation coefficient vs Frac')

	axes[0].legend()
	# axes[0].set_xlim([0,1])
	# axes[0].set_ylim([0,1])

	axes[1].plot(feat1,mse1,color='red',label='Matrix selection')
	axes[1].scatter(feat1,mse1, color='red', marker='o')

	axes[1].plot(feat2,mse2,color='blue',label='Random selection')
	axes[1].scatter(feat2,mse2, color='blue', marker='o')

	# axes[1].plot(feat3,mse3,color='green',label='dropout')
	# axes[1].scatter(feat3,mse3, color='green', marker='o')

	axes[1].set_xlabel('Fraction of samples selected')
	axes[1].set_ylabel('MSE')
	axes[1].set_title('MSE vs Frac')

	axes[1].legend()
	# axes[1].set_xlim([0,1])
	# axes[1].set_ylim([0,1])

	plt.show()

	plt.suptitle(f'{args.dataset} {args.trait}')
	plt.savefig(f'results/{args.dataset}/{args.trait}/superimpose.pdf')