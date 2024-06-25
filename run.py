import shapleyestimators as se

num_runs = 10
dataset = 'Communities'
sample_sizes = [300, 500, 1000, 2000, 5000, 10000]

results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes)

image_filename = f'images/{dataset}.pdf'

exclude= ['Tree SHAP', 'Recycled Sampling', 'Uniform Sampling Adjusted', 'Uniform Sampling Adjusted 2x', 'Uniform Sampling Sum']

for name in se.estimators:
    if 'Offset' in name:
        exclude.append(name)

se.plot_data(results, dataset, image_filename, exclude=exclude)