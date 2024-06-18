import shapleyestimators as se

num_runs = 10
dataset = 'Communities'
sample_sizes = [250, 500, 1000, 2000, 5000, 10000]

results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes)

image_filename = f'images/{dataset}.pdf'

se.plot_data(results, dataset, image_filename)