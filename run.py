import shapleyestimators as se

num_runs = 10
dataset = 'Communities'

small_n = ['Adult', 'California', 'Diabetes']

big_n = ['Communities', 'Correlated', 'Independent', 'NHANES']

m = 10000
for n in [10, 100, 1000]:
    se.plot_probs(n, folder='images/')
    se.plot_sampled_sizes(n, m, folder='images/')

for dataset in small_n + big_n:
    se.visualize_predictions(dataset, folder='images/')

for dataset in small_n + big_n:
    print(dataset)
    n = se.get_dataset_size(dataset)
    sample_sizes = [int(n * i) for i in [5, 10, 20, 40, 80, 160]]
    weighted_error = 2**n <= 1e7
    results = se.benchmark(num_runs, dataset, se.estimators, sample_sizes, weighted_error=weighted_error)

    image_filename = f'images/{dataset}.pdf'

    se.plot_data(results, dataset, image_filename, weighted_error=weighted_error)