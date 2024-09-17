import leverageshap as ls

ablation_estimators = ['Kernel SHAP', 'Official Kernel SHAP', 'Leverage SHAP', 'Kernel SHAP Paired', 'Leverage SHAP wo Paired', 'Leverage SHAP wo Bernoulli, Paired']

main_estimators = ['Kernel SHAP', 'Official Kernel SHAP', 'Leverage SHAP']

ns = [10, 12, 14, 16]
sample_size = 500
num_runs = 10
for n in ns:
    continue
    ls.benchmark_gamma(num_runs, n, ablation_estimators, sample_size=sample_size, silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': sample_size, 'noise': 0}
    results = ls.load_results([f'Synthetic_{n}' for n in ns], x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=main_estimators)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=ablation_estimators)