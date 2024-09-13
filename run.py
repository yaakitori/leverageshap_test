import leverageshap as ls

small_n = ['California', 'Diabetes', 'Adult', 'Synthetic']

big_n = ['Correlated', 'Independent', 'NHANES', 'Communities']

def get_hyperparameter_values(name):
    if name == 'noise_std':
        return [0, .5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
    elif name == 'sample_size':
        return [500, 1000, 2000, 4000, 8000, 16000]
    else:
        raise ValueError(f'Unknown hyperparameter {name}')

# Debugging access to official SHAP estimators
#import logging
#log = logging.getLogger('shap')
#logging.basicConfig(level=logging.DEBUG)

if False:

    m = 1000
    for n in [10, 100, 1000]:
        ls.plot_probs(n, folder='images/')
        ls.plot_sampled_sizes(n, m, folder='images/')

    for dataset in small_n + big_n:
        ls.visualize_predictions(dataset, folder='images/', exclude=['Official Tree SHAP'])

    num_runs = 100
    for dataset in small_n + big_n:
        print(dataset)
        for hyperparameter in ['sample_size', 'noise_std']:
            print(hyperparameter)
            ls.benchmark(num_runs, dataset, ls.estimators, hyperparameter, get_hyperparameter_values(hyperparameter), silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by number of samples
    x_name = 'sample_size'
    constraints = {'noise': 0}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error')

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 1000}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error')

    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': 1000, 'noise': 0}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=False, log_y=y_name == 'shap_error')

# Tables
for y_name in ['shap_error', 'weighted_error']:
    results = ls.load_results(small_n + big_n, 'sample_size', y_name, {'noise': 0, 'sample_size' : 2000})
    ls.one_big_table(results, f'tables/{y_name}.tex')
    #for dataset in results:
    #    ls.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)