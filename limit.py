import numpy as np
import itertools
import math

n = 6

# set the seed
gen = np.random.default_rng(2)

v = gen.random(2**n)
v[0] = 0

# Shapley values
def compute_shapley_values(n, v):
    shap_values = np.zeros(n)
    for i in range(n):
        valid_items = [j for j in range(n) if j != i]
        for set_size in range(n):
            # Generate all subsets of size set_size
            weight = 1 / math.comb(n - 1, set_size)
            for subset in itertools.combinations(valid_items, set_size):
                # All indices from 0 to 2**n correspond to a subset
                # Convert from a subset to the associated index
                index_wo_i = sum([2**j for j in subset])
                index_w_i = index_wo_i + 2**i
                shap_values[i] += weight * (v[index_w_i] - v[index_wo_i])
    return shap_values / n

shap_values = compute_shapley_values(n, v)
print(shap_values)
#print()

non_identity = np.zeros((2**n, 2**n))

for row_idx in range(2**n):
    subset = [j for j in range(n) if (row_idx >> j) & 1]
    not_in_subset = [j for j in range(n) if j not in subset]
    non_identity[row_idx, row_idx] = -1 * len(not_in_subset)

    for j in not_in_subset:
        subset_with_j = subset + [j]
        subset_with_j.sort()
        subset_with_j_idx = sum([2**i for i in subset_with_j])
        non_identity[row_idx, subset_with_j_idx] = 1
        non_identity[row_idx, 2**j] = -1

def get_single_values(resulting_v, n):
    single = np.zeros(n)
    for i in range(n):
        index = 2**i
        single[i] = resulting_v[index]
    return single

tau = 1/n
H = np.eye(2**n) + tau * non_identity

# Compute the limit
next_v = v
next_H = H
for j in range(50):
    next_v = next_H @ next_v

#print(next_v)
limit_values = get_single_values(next_v, n)
print(limit_values)
print()
print()

# Approach without matrices

gen = np.random.default_rng(5)

v = {}

for idx in range(2**n):
    subset = tuple([j for j in range(n) if (idx >> j) & 1])
    v[subset] = gen.random()
    if len(subset) == 0: v[subset] = 0

#n = 2
#v = {(): 0, (0,): 1, (1,): 3, (0, 1): 5}

def compute_next_v(v, n, epsilon):
    next_v = {}
    for idx in range(2**n):
        subset = tuple([j for j in range(n) if (idx >> j) & 1])
        not_in_subset = [j for j in range(n) if j not in subset]
        summation = 0
        for j in not_in_subset:
            subset_with_j = list(subset) + [j]
            subset_with_j.sort()
            subset_with_j = tuple(subset_with_j)
            j = tuple([j])
            summation += v[subset_with_j] - v[j] - v[subset]

        next_v[subset] = v[subset] + epsilon * summation
    return next_v


def compute_shapley_values_dict(n, v):
    shap_values = np.zeros(n)
    for i in range(n):
        valid_items = [j for j in range(n) if j != i]
        for set_size in range(n):
            # Generate all subsets of size set_size
            weight = 1 / math.comb(n - 1, set_size)
            for subset in itertools.combinations(valid_items, set_size):
                subset = tuple(subset)
                subset_wo_i = subset
                subset_w_i = list(subset) + [i]
                subset_w_i.sort()
                subset_w_i = tuple(subset_w_i)
                shap_values[i] += weight * (v[subset_w_i] - v[subset_wo_i])
    return shap_values / n

def extract_single_values(v, n):
    single = np.zeros(n)
    for i in range(n):
        single[i] = v[tuple([i])]
    return single

print(compute_shapley_values_dict(n, v))
epsilon = 1/(2*n)
next_v = v
for i in range(100):
    next_v = compute_next_v(next_v, n, tau)
print(extract_single_values(next_v, n))

        
        