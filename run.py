import shapleyestimators as se

num_runs = 10
dataset = 'Communities'
sample_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
sample_sizes = [int(s) for s in sample_sizes]

#for dataset in ['Adult', 'California', 'Communities']:
results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes)

image_filename = f'images/{dataset}.pdf'

exclude= ['Tree SHAP', 'Recycled Sampling', 'Uniform Sampling Adjusted', 'Uniform Sampling Adjusted 2x', 'Uniform Sampling Sum']

for name in se.estimators:
    if 'Offset' in name:
        exclude.append(name)

se.plot_data(results, dataset, image_filename, exclude=exclude)