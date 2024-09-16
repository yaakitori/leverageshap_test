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

    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P
    inputs = baseline * (1 - binary_Z) + explicand * binary_Z
    v1 = model.predict(explicand)
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    y = (vz - v0) / inv_sqrt_weights
    b = y - Z.sum(axis=1) * (v1 - v0) / n
    print('b', b)
    print('v1', v1)
    print('v0', v0)
    return {'A': A, 'b': b}

def get_dataset_size(dataset):
    if 'Synthetic_' in dataset:
        return int(dataset.split('_')[1])
    X, y = load_dataset(dataset)
    return X.shape[1]

def read_file(dataset, estimator, x_name, y_name, constraints={}):
    filename = f'output/{dataset}_{estimator}.csv'
    if not os.path.exists(filename): return {}
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
        self.sample_count = 0

    def predict(self, X):
        self.sample_count += len(X)
        return self.model.predict(X) + np.random.normal(0, self.noise_std, X.shape[0])
    
    def get_sample_count(self):
        return self.sample_count

def run_small_setup(baseline, explicand, model, true_shap_values):
    linear_system = build_full_linear_system(baseline, explicand, model)
    best_weighted_error = np.sum((linear_system['A'] @ true_shap_values - linear_system['b'])**2)
    Aphi = linear_system['A'] @ true_shap_values
    gamma = np.sum((Aphi - linear_system['b'])**2) / np.sum((Aphi)**2)    
    normalized_gamma = gamma / np.sum((true_shap_values)**2)
    # Round to 2 significant figures
    normalized_gamma = float(f'{normalized_gamma:.2g}')
    return {'A': linear_system['A'], 'b': linear_system['b'], 'best_weighted_error': best_weighted_error, 'normalized_gamma': normalized_gamma, 'gamma': gamma}

def run_one_iteration(X, seed, dataset, model, sample_size, noise_std, num_runs):
    baseline, explicand = load_input(X, seed=seed, is_synthetic=dataset=='Synthetic')
    n = X.shape[1]
    is_small = 2**n <= 1e7
    # Compute the true SHAP values (assuming tree model)
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, sample_size).flatten()

    small_setup = {}
     
    for estimator_name, estimator in estimators.items():        
        if estimator_name in ['Official Tree SHAP']:
            continue

        results = read_file(dataset, estimator_name, 'sample_size', 'shap_error', {'noise': noise_std, 'n': n})
        if results != {}:
            if len(results[sample_size]) >= num_runs: continue
        noised_model = NoisyModel(model, noise_std)
        shap_values = estimator(baseline, explicand, noised_model, sample_size).flatten()

        filename = f'output/{dataset}_{estimator_name}.csv'

        with open(filename, 'a') as f:
            dict = {
                'sample_size': sample_size,
                'difference': noised_model.get_sample_count() - sample_size,
                'noise': noise_std,
                'n' : n,
            }
            dict['shap_error'] = ((shap_values - true_shap_values) ** 2).mean()
            if is_small:
                if small_setup == {}:
                    small_setup = run_small_setup(baseline, explicand, model, true_shap_values)
                weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                dict['weighted_error'] = weighted_error / small_setup['best_weighted_error'] 
                dict['gamma'] = small_setup['normalized_gamma']
            f.write(str(dict) + '\n')

def benchmark(num_runs, dataset, estimators, hyperparameter, hyperparameter_values, silent=False):              

    X, y = load_dataset(dataset)
    n = X.shape[1]
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    config = {'sample_size': 1000, 'noise_std' : 0}
    for run_idx in tqdm(range(num_runs), disable=silent):
        for hyperparameter_value in hyperparameter_values:
            config[hyperparameter] = hyperparameter_value
            run_one_iteration(X, run_idx * num_runs, dataset, model, sample_size=config['sample_size'], noise_std=config['noise_std'], num_runs=num_runs)

class SyntheticModel:
    def __init__(self, v, correspondence):
        self.v = v
        self.num_samples = 0
        self.correspondence = correspondence

    def predict(self, X):
        # X is a binary matrix
        # Get the integer represented in each row
        indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
        # Get the corresponding index in v
        self.num_samples += len(X)
        return self.v[[self.correspondence[i] for i in indices]]
    
    def get_sample_count(self):
        return self.num_samples

def build_gamma_labels(n, alpha):
    # Construct A
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    # Convert all rows to their integer form
    X = binary_Z
    indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
    correspondence = {0:0, 2**n-1:-1}
    for i in range(2**n-2):
        correspondence[indices[i]] = i
    print('indices', indices)
    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))
    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P

    # Perform QR decomposition of A
    Q, R = np.linalg.qr(A)

    # The last column of Q is orthogonal to all the columns of A
    # if A has full rank and is not square
    r = Q[:, -1]
    
    # Check that r is orthogonal to the columns of A
    assert np.allclose(A.T @ r, 0)

    # Construct b as (1-alpha) * a column of A + alpha * r
    normalized_first_col = A[:, 0] / np.linalg.norm(A[:, 0])
    b = (1 - alpha) * normalized_first_col + alpha * r

    # Convert from b to y
    v1 = 1
    v0 = 0
    y = b + Z.sum(axis=1) * (v1 - v0) /n 

    v = np.zeros(2**n)
    v[1:-1] = y * inv_sqrt_weights
    v[0] = v0
    v[-1] = v1

    # True SHAP values.
    # Solve Ax = b
    true_shap_values = np.linalg.lstsq(A, b, rcond=None)[0]
    best_weighted_error = np.sum((A @ true_shap_values - b)**2)

    gamma = np.sum((A @ true_shap_values - b)**2) / np.sum((A @ true_shap_values)**2)
    print(f'Gamma: {gamma}')
    print('b', b)

    return {'v': v, 'true_shap_values': true_shap_values, 'best_weighted_error': best_weighted_error, 'correspondence': correspondence}

def benchmark_gamma(num_runs, n, estimators, silent=False):
    sample_size = 1000
    baseline = np.zeros((1, n))
    explicand = np.ones((1, n))
    for run_idx in tqdm(range(num_runs), disable=silent):
        for alpha in [.2, .3, .4, .5, .6, .7, .8]:
            seed = run_idx * num_runs + int(alpha * 100)
            np.random.seed(seed)
            gamma_labels = build_gamma_labels(n, alpha)

            is_small = 2**n <= 1e7

            small_setup = {}

            dataset = 'Synthetic_' + str(n)
            
            for estimator_name, estimator in estimators.items():
                if estimator_name in ['Official Tree SHAP', 'Official Kernel SHAP']:
                    continue
                model = SyntheticModel(gamma_labels['v'], gamma_labels['correspondence'])
                results = read_file(dataset, estimator_name, 'alpha', 'shap_error', {'n': n})
                if results != {} and alpha in results:
                    if len(results[alpha]) >= num_runs: continue
                shap_values = estimator(baseline, explicand, model, sample_size).flatten()

                filename = f'output/{dataset}_{estimator_name}.csv'

                with open(filename, 'a') as f:
                    dict = {
                        'sample_size': sample_size,
                        'difference': model.get_sample_count() - sample_size,
                        'noise': 0,
                        'n' : n,
                        'alpha' : alpha,
                    }
                    dict['shap_error'] = ((shap_values - gamma_labels['true_shap_values']) ** 2).mean()
                    if is_small:
                        if small_setup == {}:
                            small_setup = run_small_setup(baseline, explicand, model, gamma_labels['true_shap_values'])
                        weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                        dict['weighted_error'] = weighted_error / gamma_labels['best_weighted_error'] 
                        dict['gamma'] = small_setup['normalized_gamma']
                    f.write(str(dict) + '\n')