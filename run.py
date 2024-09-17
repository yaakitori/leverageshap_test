import leverageshap as ls

small_n = ['IRIS', 'California', 'Diabetes', 'Adult']

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

ablation_estimators = ['Kernel SHAP', 'Official Kernel SHAP', 'Leverage SHAP', 'Kernel SHAP Paired', 'Leverage SHAP wo Paired', 'Leverage SHAP wo Bernoulli, Paired']

main_estimators = ['Kernel SHAP', 'Official Kernel SHAP', 'Leverage SHAP']

datasets = small_n + big_n

if True:

    ls.plot_probs([10,100,1000], folder='images/')

    ls.visualize_predictions(datasets, main_estimators, filename='images/main_detailed.pdf')
    ls.visualize_predictions(datasets, ablation_estimators, filename='images/ablation_detailed.pdf')

    num_runs = 10
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
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators)

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 1000}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators)

# Tables
for y_name in ['shap_error', 'weighted_error']:
    results = ls.load_results(small_n + big_n, 'sample_size', y_name, {'noise': 0, 'sample_size' : 2000})
    ls.one_big_table(results, f'tables/{y_name}.tex')
    #for dataset in results:
    #    ls.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)