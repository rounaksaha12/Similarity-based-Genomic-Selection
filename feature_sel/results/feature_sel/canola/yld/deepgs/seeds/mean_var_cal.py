import numpy as np
import pandas as pd

if __name__ == '__main__':
	seeds = [354, 694, 1446, 2234, 2443, 2716]
    
	seed_dict = {}

	for seed in seeds:
		df = pd.read_csv(f'deepgs_seed_{seed}.csv')

		for _, row in df.iterrows():
			if row['Cols retained'] in seed_dict.keys():
				seed_dict[row['Cols retained']]['pearson'].append(row['pearson_r'])
				seed_dict[row['Cols retained']]['mse'].append(row['mse'])
			else:
				seed_dict[row['Cols retained']] = {'pearson': [row['pearson_r']], 'mse': [row['mse']]}
	# print(seed_dict)
	print('Columns retained, Pearson mean, Pearson std, MSE mean, MSE std')
	for cols_retained in sorted(seed_dict.keys()):
		pearson_vals = np.array(seed_dict[cols_retained]['pearson'])
		mse_vals = np.array(seed_dict[cols_retained]['mse'])

		print(f'{cols_retained}, {np.mean(pearson_vals)}, {np.std(pearson_vals)}, {np.mean(mse_vals)}, {np.std(mse_vals)}')
