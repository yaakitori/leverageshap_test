import leverageshap as ls

ablation_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP', 'Kernel SHAP Paired', 'Leverage SHAP wo Bernoulli', 'Leverage SHAP wo Bernoulli, Paired']

main_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP']

include_estimators = ['Leverage SHAP', 'Leverage SHAP wo Paired']

ns = [8, 10, 12, 14]
num_runs = 100
if True:
    for n in ns:
        ls.benchmark_gamma(num_runs, n, include_estimators, sample_size=10*n, silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by gamma
    x_name = 'alpha'
    constraints = {'noise': 0}
    results = ls.load_results([f'Synthetic_{n}' for n in ns], x_name, y_name, constraints, is_actual_sample_size=True)

    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_gamma_{y_name}.pdf', log_x=False, log_y=True, include_estimators=include_estimators, plot_mean=False)

    #ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=ablation_estimators, plot_mean=True)