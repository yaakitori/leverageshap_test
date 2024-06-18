from .estimators import *
import numpy as np
import xgboost as xgb
import shap
import os
from tqdm import tqdm

dataset_loaders = {
    'Communities' : shap.datasets.communitiesandcrime,
}

def read_file(dataset, estimator):
    filename = f'output/{dataset}_{estimator}.csv'
    saved = {}
    if not os.path.exists(filename):
        return saved
    with open(filename, 'r') as f:
        for line in f:
            dict = eval(line)
            if dict['sample_size'] not in saved:
                saved[dict['sample_size']] = []
            saved[dict['sample_size']].append(dict['error'])
    return saved

def benchmark(num_runs, dataset, estimators, sample_sizes = [1000], silent=False):
    X, y = dataset_loaders[dataset]()
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    saved = {}
    for estimator_name in estimators.keys():
        saved[estimator_name] = read_file(dataset, estimator_name)
        for sample_size in sample_sizes:
            if sample_size not in saved[estimator_name]:
                saved[estimator_name][sample_size] = []

    for run_idx in tqdm(range(num_runs), disable=silent):
        for sample_size in sample_sizes:            
            # Randomly choose a baseline and explicand
            np.random.seed(run_idx * num_runs + sample_size)
            baseline_idx = np.random.choice(X.shape[0])
            explicand_idx = np.random.choice(X.shape[0])
            baseline = X.iloc[baseline_idx].values.reshape(1, -1)
            explicand = X.iloc[explicand_idx].values.reshape(1, -1)

            # Compute the true SHAP values (assuming tree model)
            true_shap_values = estimators['Tree SHAP'](baseline, explicand, model, sample_size)

            for estimator_name, estimator in estimators.items():
                if len(saved[estimator_name][sample_size]) >= num_runs:
                    continue
                shap_values = estimator(baseline, explicand, model, sample_size)
                error = ((shap_values - true_shap_values) ** 2).mean()
                filename = f'output/{dataset}_{estimator_name}.csv'
                with open(filename, 'a') as f:
                    dict = {'sample_size': sample_size, 'error': error}
                    f.write(str(dict) + '\n')
    
    saved = {}
    for estimator_name in estimators.keys():
        saved[estimator_name] = read_file(dataset, estimator_name)
    return saved



