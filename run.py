import shapleyestimators as se

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

    m = 10000
    for n in [10, 100, 1000]:
        se.plot_probs(n, folder='images/')
        se.plot_sampled_sizes(n, m, folder='images/')

    for dataset in small_n + big_n:
        se.visualize_predictions(dataset, folder='images/')

    num_runs = 0
    for dataset in small_n + big_n:
        print(dataset)
        for hyperparameter in ['sample_size', 'noise_std']:
            print(hyperparameter)
            se.benchmark(num_runs, dataset, se.estimators, hyperparameter, get_hyperparameter_values(hyperparameter), silent=False)

for y_name in ['shap_error', 'weighted_error']:
    # Performance by number of samples
    x_name = 'sample_size'
    constraints = {'noise': 0}
    results = se.load_results(small_n + big_n, x_name, y_name, constraints)
    se.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error')

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 4000}
    results = se.load_results(small_n + big_n, x_name, y_name, constraints)
    se.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error')

    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': 1000, 'noise': 0}
    results = se.load_results(small_n + big_n, x_name, y_name, constraints)
    se.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=False, log_y=y_name == 'shap_error')

# Plotting
# Performance by number of samples
# Performance by noise level
# Performance by gamma

    #if 2**n <= 1e7:
    #    results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes, weighted_error=True, verbose=False)
    #    image_filename = f'images/{dataset}_weighted.pdf'
    #    se.plot_data(results, dataset, image_filename, weighted_error=True)
    #results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes, weighted_error=False, verbose=False)
    #image_filename = f'images/{dataset}_l2.pdf'
    #se.plot_data(results, dataset, image_filename, weighted_error=False)

