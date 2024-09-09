import matplotlib.pyplot as plt
from .estimators import *
from .datasets import *
import numpy as np
import xgboost as xgb
import os
from tqdm import tqdm
import scipy

# Every line of output files contains a dictionary with the following keys
# 'sample_size': number of samples used to estimate SHAP values
# 'noise': standard deviation of noise added to the labels
# 'shap_error': mean squared error between estimated and true SHAP values
# 'weighted_error' (optional): ||Ax- b||^2 / ||Ax* - b||^2 where x* is the true SHAP values and x is the estimated SHAP values
# 'gamma' (optional): ||b||^2 / ||Ax||^2 where x is the estimated SHAP values

def build_full_linear_system(baseline, explicand, model):
    n = baseline.shape[1]
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))
    # Error in the following line: ValueError: operands could not be broadcast together with shapes (4094,) (4094,12) 
#    Z = 1 / inv_sqrt_weights * binary_Z
    # Fix the error by changing the following line
    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P
    inputs = baseline * (1 - binary_Z) + explicand * binary_Z
    v1 = model.predict(explicand)
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    y = (vz - v0) / inv_sqrt_weights
    b = y - Z.sum(axis=1) * (v1 - v0) / n
    return {'A': A, 'b': b}

def get_dataset_size(dataset):
    X, y = load_dataset(dataset)
    return X.shape[1]

def read_file(dataset, estimator, x_name, y_name, constraints={}):
    filename = f'output/{dataset}_{estimator}.csv'
    results = {}
    with open(filename, 'r') as f:
        for line in f:
            dict = eval(line)
            add = True
            for key, value in constraints.items():
                if dict[key] != value:
                    add = False
            if add:
                try:
                    x, y = dict[x_name], dict[y_name]
                    if x not in results:
                        results[x] = []
                    results[x].append(y)
                except KeyError:
                    pass
    return results

def load_results(datasets, x_name, y_name, constraints, estimator_names=estimators.keys()):
    results_by_dataset = {}
    for dataset in datasets:
        results_by_estimator = {}
        for estimator_name in estimator_names:
            if estimator_name == 'Official Tree SHAP':
                continue
            results = read_file(dataset, estimator_name, x_name, y_name, constraints)
            if results != {}:
                results_by_estimator[estimator_name] = results
        if results_by_estimator != {}:
            results_by_dataset[dataset] = results_by_estimator
    return results_by_dataset

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

def visualize_predictions(dataset, folder='', exclude=[]):
    plt.clf()
    X, y = load_dataset(dataset)
    n = X.shape[1]
    num_samples = 5 * n
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    baseline, explicand = load_input(X)
    # 2 by 3 array of axes in matplotlib plot
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, num_samples).flatten()
    i = 0
    for estimator_name, estimator in estimators.items():
        if estimator_name in exclude:
            continue
        shap_values = estimator(baseline, explicand, model, num_samples).flatten()
        m, b = np.polyfit(true_shap_values, shap_values, 1)
        ax = axs[i // 3, i % 3]
        ax.plot(true_shap_values, m * true_shap_values + b, color='green', linewidth=1, alpha=0.5)
        ax.scatter(true_shap_values, shap_values, alpha=0.5, marker='.')
        ax.set_title(estimator_name)
        i += 1
    
    # Set title for whole plot
    fig.suptitle(rf'{dataset} Dataset ($n = {n}$)', fontsize=20)
    # Set x label for bottom row
    for ax in axs[1]:
        ax.set_xlabel(r'True Shapley Values ($\phi$)')
    # Set y label for left column
    for ax in axs[:,0]:
        ax.set_ylabel(r'Predicted Shapley Values ($\tilde{\phi}$)')     

    filename = f'{folder}/{dataset}_detailed.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()

class NoisyModel:
    def __init__(self, model, noise_std):
        self.model = model
        self.noise_std = noise_std
    def predict(self, X):
        return self.model.predict(X) + np.random.normal(0, self.noise_std, X.shape[0])

def run_one_iteration(X, seed, dataset, model, sample_size, noise_std):
    baseline, explicand = load_input(X, seed=seed, is_synthetic=dataset=='Synthetic')
    n = X.shape[1]
    is_small = 2**n <= 1e7
    # Compute the true SHAP values (assuming tree model)
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, sample_size).flatten()

    if is_small:
        linear_system = build_full_linear_system(baseline, explicand, model)
        best_weighted_error = np.sum((linear_system['A'] @ true_shap_values - linear_system['b'])**2)
        gamma = np.sum(linear_system['b']**2) / np.sum((linear_system['A'] @ true_shap_values)**2)    
        # Round for plotting purposes
        gamma = round(gamma, 1)
     
    noised_model = NoisyModel(model, noise_std)
    for estimator_name, estimator in estimators.items():
        if estimator_name in ['Official Tree SHAP']:
            continue
        while True:
            try:
                shap_values = estimator(baseline, explicand, noised_model, sample_size).flatten()
                break
            except np.linalg.LinAlgError:
                print('LinAlgError:', estimator_name)
                pass

        filename = f'output/{dataset}_{estimator_name}.csv'
        with open(filename, 'a') as f:
            dict = {'sample_size': sample_size, 'noise': noise_std}
            dict['shap_error'] = ((shap_values - true_shap_values) ** 2).mean()
            if is_small:
                weighted_error = np.sum((linear_system['A'] @ shap_values - linear_system['b'])**2)
                dict['weighted_error'] = weighted_error / best_weighted_error
                dict['gamma'] = gamma
            f.write(str(dict) + '\n')

def benchmark(num_runs, dataset, estimators, hyperparameter, hyperparameter_values, silent=False):              
#              sample_sizes = None, silent=False, weighted_error=False, verbose=False):
    X, y = load_dataset(dataset)
    n = X.shape[1]
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
#    error_name = 'weighted_error' if weighted_error else 'shap_error'

#    saved = {}
#    for estimator_name in estimators.keys():
#        saved[estimator_name] = read_file(dataset, estimator_name, error_name)
#        for sample_size in sample_sizes:
#            if sample_size not in saved[estimator_name]:
#                saved[estimator_name][sample_size] = []
    config = {'sample_size': 4000, 'noise_std' : 0}
    for run_idx in tqdm(range(num_runs), disable=silent):
        for hyperparameter_value in hyperparameter_values:
            config[hyperparameter] = hyperparameter_value
            run_one_iteration(X, run_idx * num_runs, dataset, model, sample_size=config['sample_size'], noise_std=config['noise_std'])
            ## Randomly choose a baseline and explicand
            ## Choose baseline and explicand so no variables are the same
            #baseline, explicand = load_input(X, seed=run_idx * num_runs, is_synthetic=dataset=='Synthetic')

            ## Compute the true SHAP values (assuming tree model)
            #true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, sample_size)
            #if weighted_error:
            #    best_weighted_error = compute_weighted_error(baseline, explicand, model, true_shap_values)

            #for estimator_name, estimator in estimators.items():
            #    if len(saved[estimator_name][sample_size]) >= num_runs:
            #        continue
            #    while True:
            #        try:
            #            shap_values = estimator(baseline, explicand, model, sample_size)
            #            break
            #        except np.linalg.LinAlgError:
            #            pass
            #    if verbose:
            #        print(estimator_name)
            #        print('shap values', shap_values)
            #        print('true shap values', true_shap_values)
            #        print('explicand - baseline', (explicand - baseline))

            #    filename = f'output/{dataset}_{estimator_name}.csv'
            #    with open(filename, 'a') as f:
            #        dict = {'sample_size': sample_size}
            #        dict['shap_error'] = ((shap_values - true_shap_values) ** 2).mean()
            #        if weighted_error:
            #            dict['weighted_error'] = compute_weighted_error(baseline, explicand, model, shap_values) / best_weighted_error
            #        f.write(str(dict) + '\n')
    
    #saved = {'n': X.shape[1]}
    #for estimator_name in estimators.keys():
    #    saved[estimator_name] = read_file(dataset, estimator_name, 'shap_error')
    #return saved



