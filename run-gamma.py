import leverageshap as ls

small_n = ['California', 'Diabetes', 'Adult', 'Synthetic']


ns = [10]
num_runs = 10
for n in ns:
    print(n)
    ls.benchmark_gamma(num_runs, n, ls.estimators, silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': 1000, 'noise': 0}
    results = ls.load_results([f'Synthetic_{n}' for n in ns], x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/{x_name}-{y_name}.pdf', log_x=True, log_y=False)