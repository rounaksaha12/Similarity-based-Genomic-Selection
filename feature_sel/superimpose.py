import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse

def extract(filename):
	df = pd.read_csv(filename)
	feat = np.array(df['Cols retained'])
	pearson = np.array(df['pearson_r'])
	mse = np.array(df['mse'])
	return feat, pearson, mse


if __name__=='__main__':


	# downloading font for rendering matplotlib text
	fm.fontManager.addfont('./fontdir/LibreBaskerville-Regular.ttf')
	mpl.rc('font', family='Libre Baskerville')

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--trait', type=str)

	args = parser.parse_args()

	filedir = f'results/feature_sel/{args.dataset}/{args.trait}/'


	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
	plt.subplots_adjust(wspace=0.2,hspace=0.3)

	feat1, pearson1, mse1 = extract(filedir+'obscured/obscured.csv')
	feat2, pearson2, mse2 = extract(filedir+'deepgs/deepgs.csv')
	feat3, pearson3, mse3 = extract(filedir+'ridge/ridge.csv')
	feat4, pearson4, mse4 = extract(filedir+'rf/rf.csv')

	axes[0].plot(feat1,pearson1,color='red',label='Obscured Model')
	axes[0].scatter(feat1,pearson1, color='red', marker='o')

	axes[0].plot(feat2,pearson2,color='blue',label='DeepGS')
	axes[0].scatter(feat2,pearson2, color='blue', marker='o')

	axes[0].plot(feat3,pearson3,color='green',label='Ridge regressor')
	axes[0].scatter(feat3,pearson3, color='green', marker='o')

	axes[0].plot(feat4,pearson4,color='brown',label='Random Forest')
	axes[0].scatter(feat4,pearson4, color='brown', marker='o')

	axes[0].set_xlabel('Number of Features')
	axes[0].set_ylabel('Pearson correlation coefficient')
	axes[0].set_xscale('log')
	axes[0].set_title('Pearson correlation coefficient vs Number of Features')

	axes[0].legend()

	axes[1].plot(feat1,mse1,color='red',label='Obscured Model')
	axes[1].scatter(feat1,mse1, color='red', marker='o')

	axes[1].plot(feat2,mse2,color='blue',label='DeepGS')
	axes[1].scatter(feat2,mse2, color='blue', marker='o')

	axes[1].plot(feat3,mse3,color='green',label='Ridge regressor')
	axes[1].scatter(feat3,mse3, color='green', marker='o')

	axes[1].plot(feat4,mse4,color='brown',label='Random Forest')
	axes[1].scatter(feat4,mse4, color='brown', marker='o')

	axes[1].set_xlabel('Number of Features')
	axes[1].set_ylabel('MSE')
	axes[1].set_xscale('log')
	axes[1].set_title('MSE vs Number of Features')

	axes[1].legend()

	plt.suptitle(f'{args.dataset} {args.trait}')
	plt.savefig(f'results/feature_sel/{args.dataset}/{args.trait}/superimpose.pdf')