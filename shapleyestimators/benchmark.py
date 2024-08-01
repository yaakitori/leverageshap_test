import matplotlib.pyplot as plt
from .estimators import *
import numpy as np
import xgboost as xgb
import shap
import os
from tqdm import tqdm
import scipy

dataset_loaders = {
    'Adult' : shap.datasets.adult,
    'California' : shap.datasets.california,
    'Communities' : shap.datasets.communitiesandcrime,
    'Correlated' : shap.datasets.corrgroups60,
    'Diabetes' : shap.datasets.diabetes,
    'Independent' : shap.datasets.independentlinear60,
    'IRIS' : shap.datasets.iris,
    'NHANES' : shap.datasets.nhanesi,
}

def get_dataset_size(dataset):
    X, y = dataset_loaders[dataset]()
    return X.shape[1]

def read_file(dataset, estimator, error_name):
    filename = f'output/{dataset}_{estimator}.csv'
    saved = {}
    if not os.path.exists(filename):
        return saved
    with open(filename, 'r') as f:
        for line in f:
            dict = eval(line)            
            if dict['sample_size'] not in saved:
                saved[dict['sample_size']] = []
            saved[dict['sample_size']].append(dict[error_name])                
    return saved

def compute_weighted_error(baseline, explicand, model, shap_values):
    n = baseline.shape[1]
    Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            Z[idx, list(indices)] = 1
            idx += 1
    Z1_norm = np.sum(Z, axis=1)
    inv_weights = Z1_norm * (n - Z1_norm) * scipy.special.binom(n, Z1_norm)
    weights = 1 / inv_weights
    inputs = baseline * (1 - Z) + explicand * Z
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    return np.sum(weights * (shap_values @ Z.T - (vz - v0)) ** 2)

def load_input(X, seed=None):
    if seed is not None:
        np.random.seed(seed)
    baseline = X.mean().values.reshape(1, -1)
    explicand_idx = np.random.choice(X.shape[0])
    explicand = X.iloc[explicand_idx].values.reshape(1, -1)
    for i in range(explicand.shape[1]):
        while baseline[0, i] == explicand[0, i]:
            explicand_idx = np.random.choice(X.shape[0])
            explicand[0,i] = X.iloc[explicand_idx, i]
    return baseline, explicand

def visualize_predictions(dataset, folder=''):
    X, y = dataset_loaders[dataset]()
    n = X.shape[1]
    num_samples = 5 * n
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    baseline, explicand = load_input(X)
    # 2 by 3 array of axes in matplotlib plot
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, num_samples).flatten()
    for i, (estimator_name, estimator) in enumerate(estimators.items()):     
        shap_values = estimator(baseline, explicand, model, num_samples).flatten()
        m, b = np.polyfit(true_shap_values, shap_values, 1)
        ax = axs[i // 3, i % 3]
        ax.plot(true_shap_values, m * true_shap_values + b, color='green', linestyle='dashed', linewidth=1, alpha=0.5)
        ax.scatter(true_shap_values, shap_values, alpha=0.5, marker='.')
        ax.set_title(estimator_name)
    
    # Set title for whole plot
    fig.suptitle(f'{dataset} Dataset')
    # Set x label for bottom row
    for ax in axs[1]:
        ax.set_xlabel('True SHAP Values')
    # Set y label for left column
    for ax in axs[:,0]:
        ax.set_ylabel('Predicted SHAP Values')     

    filename = f'{folder}/{dataset}_detailed.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()

def benchmark(num_runs, dataset, estimators, sample_sizes = None, silent=False, weighted_error=False):
    X, y = dataset_loaders[dataset]()
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    error_name = 'weighted_error' if weighted_error else 'shap_error'

    saved = {}
    for estimator_name in estimators.keys():
        saved[estimator_name] = read_file(dataset, estimator_name, error_name)
        for sample_size in sample_sizes:
            if sample_size not in saved[estimator_name]:
                saved[estimator_name][sample_size] = []

    for run_idx in tqdm(range(num_runs), disable=silent):
        for sample_size in sample_sizes:            
            # Randomly choose a baseline and explicand
            # Choose baseline and explicand so no variables are the same
            baseline, explicand = load_input(X, seed=run_idx * num_runs)

            # Compute the true SHAP values (assuming tree model)
            true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, sample_size)
            if weighted_error:
                best_weighted_error = compute_weighted_error(baseline, explicand, model, true_shap_values)

            for estimator_name, estimator in estimators.items():
                if len(saved[estimator_name][sample_size]) >= num_runs:
                    continue
                while True:
                    try:
                        shap_values = estimator(baseline, explicand, model, sample_size)
                        break
                    except np.linalg.LinAlgError:
                        pass
                if False:
                    print(estimator_name)
                    print('shap values', shap_values)
                    print('true shap values', true_shap_values)
                    print('explicand - baseline', (explicand - baseline).round(2))

                filename = f'output/{dataset}_{estimator_name}.csv'
                with open(filename, 'a') as f:
                    dict = {'sample_size': sample_size}
                    dict['shap_error'] = ((shap_values - true_shap_values) ** 2).mean()
                    if weighted_error:
                        dict['weighted_error'] = compute_weighted_error(baseline, explicand, model, shap_values) / best_weighted_error
                    f.write(str(dict) + '\n')
    
    saved = {'n': X.shape[1]}
    for estimator_name in estimators.keys():
        saved[estimator_name] = read_file(dataset, estimator_name, error_name)
    return saved



