import scipy.special
from .kernel import kernel_shap
from .sampling import shapley_sampling
from .tree import tree_shap
from .permutation import permutation_shap
from .complementary import complementary_contribution

import numpy as np
import xgboost as xgb
import scipy

def embedded_lattice(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    gen = np.random.Generator(np.random.PCG64())
    n = baseline.shape[1] # number of items
    # k is the number of items in a group    
    # number of function calls = num_groups * samples_per_group
    # num_groups = n/k
    # samples_per_group = 2^k 
    num_reps = 1
    while n * num_reps <= num_samples: num_reps += 1
    num_reps = min(num_reps, 10)
    k = 1
    while n / (k+1) * 2**(k+1) * num_reps <= num_samples: k += 1    
    num_groups = (n + k - 1) // k # Round up
    # All binary strings of length k
    binary_strings = np.array([list(np.binary_repr(i, width=k)) for i in range(2**k)], dtype=int)
    # Sparse matrix of all binary strings
    edges_in_lattice = {}
    # Lookup table for binary strings
    binary_lookup = {}
    for idx, binary in enumerate(binary_strings):
        binary_lookup[tuple(binary)] = idx
    for i in range(k):
        edges_in_lattice[i] = []
        for idx, binary in enumerate(binary_strings):            
            if binary[i] == 1:
                binary_from = binary.copy()
                binary_from[i] = 0
                idx_from = binary_lookup[tuple(binary_from)]
                edges_in_lattice[i].append((idx_from, idx))
        #print(edges_in_lattice[i])
    estimates = np.zeros(n)
    nestimates = np.zeros(n)
    for _ in range(num_reps):
        # Randomly partition the n features into groups of size k
        permutation = gen.permutation(n) 
        groups = np.array_split(permutation, num_groups)
        for group in groups:
            # Ensure length of group is k
            if len(group) < k:
                remaining_items = np.setdiff1d(permutation, group)
                additional = gen.choice(remaining_items, k - len(group), replace=False)
                group = np.append(group, additional)
            # Random assignment to remaining items
            remaining_items = np.setdiff1d(permutation, group)
            set_size = gen.integers(0, len(remaining_items), endpoint=True)
            additional = gen.choice(remaining_items, set_size, replace=False)
            baseline_copy = baseline.copy()
            baseline_copy[0, additional] = explicand[0, additional]
            # Create inputs
            inputs = np.tile(baseline_copy, (2**k, 1))
            for i, binary in enumerate(binary_strings):
                #print(f'Group {group} Binary {binary}')
                #print(f'Included {group[binary.astype(bool)]}')
                included = group[binary.astype(bool)]
                inputs[i, included] = explicand[0, included]
            outputs = eval_model(inputs)
            for i in group:
                i_lattice = np.where(group == i)[0][0]
                for (idx_from, idx_to) in edges_in_lattice[i_lattice]:                
                    #print(f'Estimating {i} from {group} ({i_lattice} in group)')
                    #print(f'From {idx_from} to {idx_to}')
                    #print(f'{binary_strings[idx_from]} -> {binary_strings[idx_to]}')
                    estimates[i] += outputs[idx_to] - outputs[idx_from]
                    nestimates[i] += 1
        #print(estimates)
    #print(nestimates)
    phi = estimates / nestimates

    return phi    

def recycled_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    synth_data = np.tile(baseline, (num_samples, 1))
    chosens = np.zeros((num_samples, num_features))
    for idx in range(num_samples):
        sample_size = gen.integers(0, num_features, endpoint=True)
        chosen = gen.choice(range(num_features), sample_size, replace=False)
        chosens[idx, chosen] = 1
        synth_data[idx, chosen] = explicand[0, chosen]
    predictions = eval_model(synth_data)
    phi = np.zeros(num_features)
    for i in range(num_features):
        included_indices = np.where(chosens[:,i] == 1)[0]
        included = predictions[included_indices].mean()
        excluded_indices = np.where(chosens[:,i] == 0)[0]
        excluded = predictions[excluded_indices].mean()
        phi[i] = included - excluded
    return phi

def uniform_sampling_clever(baseline, explicand, model, num_samples):
    num_features = len(baseline[0])
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    is_changed = np.abs(baseline - explicand) > 1e-6
    varying_features = np.array(list(range(num_features)))[is_changed.squeeze()]
    phi = np.zeros((1,num_features))

    sample_stage1 = np.min([num_samples, 100*len(varying_features)])
    sample_stage2 = np.max(num_samples - sample_stage1, 0)

    num_samples = sample_stage1 // len(varying_features)
    values = np.zeros((num_samples, num_features))
    for i in varying_features:
        with_data = np.tile(baseline, (num_samples, 1))
        without_data = np.tile(baseline, (num_samples, 1))
        for idx in range(num_samples):
            chosen = np.random.choice(num_features, i, replace=False)
            with_data[idx, chosen] = explicand[0, chosen]
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        values[:,i] = with_vals - without_vals
    
    means = np.mean(values, axis=0)
    vars = np.var(values, axis=0)
    vars = vars / np.sum(vars)

    sample_allocation = (vars * sample_stage2).astype(int)

    for i in varying_features:
        num_samples2 = sample_allocation[i]
        with_data = np.tile(baseline, (num_samples2, 1))
        without_data = np.tile(baseline, (num_samples2, 1))
        for idx in range(num_samples2):
            chosen = np.random.choice(num_features, i, replace=False)
            with_data[idx, chosen] = explicand[0, chosen]
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        mean = np.mean(with_vals - without_vals)
        phi[0,i] = (means[i] * num_samples + mean * num_samples2) / (num_samples + num_samples2)
    
    return phi

def permutation_recycling(baseline, explicand, model, num_samples, offset=1):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    phi = np.zeros(num_features)
    samples_per_i = np.zeros(num_features)
    for idx in range(num_samples // num_features):
        data = np.tile(baseline, (num_features, 1))
        permutation = gen.permutation(num_features) 
        for i in range(len(permutation)):
            set_i = permutation[:i]
            data[i, set_i] = explicand[0, set_i]
        vals = eval_model(data) # num_features samples            
        for i in range(num_features):
            location = np.where(permutation == i)[0][0]
            start = max(0, location - offset)
            for j in range(start+1, location + 1):
                if j + offset >= num_features:
                    continue
                phi[i] += vals[j+offset] - vals[j]
                samples_per_i[i] += 1
    phi /= samples_per_i  
    return phi

def uniform_sampling(baseline, explicand, model, num_samples):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    num_samples_per_feature = (num_samples // num_features) // 2
    phi = np.zeros((1,num_features))
    for i in range(num_features):
        with_data = np.tile(baseline, (num_samples_per_feature, 1))
        without_data = np.tile(baseline, (num_samples_per_feature, 1))
        for idx in range(num_samples_per_feature):            
            sample_size = gen.integers(0, num_features, endpoint=True)
            chosen = gen.choice(range(num_features), sample_size, replace=False)
            if i not in chosen:
                chosen = np.append(chosen, i)
            # With i included
            with_data[idx, chosen] = explicand[0, chosen]
            if i in chosen:
                chosen = chosen[chosen != i]
            # Without i included
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        vals = with_vals - without_vals
        phi[0,i] = np.mean(vals)
    return phi

def uniform_sampling_sum(baseline, explicand, model, num_samples):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    num_samples_per_feature = (num_samples // num_features) // 2
    phi = np.zeros((1,num_features))
    vars = np.zeros((1,num_features))
    for i in range(num_features):
        with_data = np.tile(baseline, (num_samples_per_feature, 1))
        without_data = np.tile(baseline, (num_samples_per_feature, 1))
        for idx in range(num_samples_per_feature):            
            sample_size = gen.integers(0, num_features, endpoint=True)
            chosen = gen.choice(range(num_features), sample_size, replace=False)
            if i not in chosen:
                chosen = np.append(chosen, i)
            # With i included
            with_data[idx, chosen] = explicand[0, chosen]
            if i in chosen:
                chosen = chosen[chosen != i]
            # Without i included
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        vals = with_vals - without_vals
        phi[0,i] = np.mean(vals)
        vars[0,i] = np.var(vals)
    sum_error = eval_model(explicand) - eval_model(baseline) - np.sum(phi)
    vars = vars / np.max(vars)
    adj = sum_error * (vars - vars * vars.sum()) / (1 + vars.sum())
    return phi + adj

def uniform_sampling_offset(baseline, explicand, model, num_samples, offset=0):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    num_samples_per_feature = (num_samples // num_features) // 2
    phi = np.zeros((1,num_features))
    for i in range(num_features):
        with_data = np.tile(baseline, (num_samples_per_feature, 1))
        without_data = np.tile(baseline, (num_samples_per_feature, 1))
        for idx in range(num_samples_per_feature):            
            sample_size = gen.integers(0, num_features, endpoint=True)
            chosen = gen.choice(range(num_features), sample_size, replace=False)
            # Remove offset number of samples from chosen
            if offset > 0:
                chosen = chosen[:-offset]
            if i not in chosen:
                chosen = np.append(chosen, i)
            # With i included
            with_data[idx, chosen] = explicand[0, chosen]
            if i in chosen:
                chosen = chosen[chosen != i]
            # Without i included
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        phi[0,i] = np.mean(with_vals - without_vals)
    return phi

def uniform_sampling_adjusted(baseline, explicand, model, num_samples):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    num_samples_per_feature = (num_samples // num_features) // 2
    phi = np.zeros((1,num_features))
    for i in range(num_features):
        with_data = np.tile(baseline, (num_samples_per_feature, 1))
        without_data = np.tile(baseline, (num_samples_per_feature, 1))
        for idx in range(num_samples_per_feature):            
            sample_size = gen.integers(0, num_features, endpoint=True)
            chosen = gen.choice(range(num_features), sample_size, replace=False)
            if i not in chosen:
                chosen = np.append(chosen, i)
            # With i included
            with_data[idx, chosen] = explicand[0, chosen]
            if i in chosen:
                chosen = chosen[chosen != i]
            # Without i included
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        preds = with_vals - without_vals
        learned_model = xgb.XGBRegressor(n_estimators=100, max_depth=4).fit(with_data, preds)
        adjustment = learned_model.predict(with_data)
        adjustment_centered = adjustment - np.mean(adjustment)
        phi[0,i] = np.mean(with_vals - without_vals - adjustment_centered)
    return phi

def uniform_sampling_adjusted2(baseline, explicand, model, num_samples):
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    num_samples_per_feature = (num_samples // num_features) // 2
    phi = np.zeros((1,num_features))
    for i in range(num_features):
        with_data = np.tile(baseline, (num_samples_per_feature, 1))
        without_data = np.tile(baseline, (num_samples_per_feature, 1))
        for idx in range(num_samples_per_feature):            
            sample_size = gen.integers(0, num_features, endpoint=True)
            chosen = gen.choice(range(num_features), sample_size, replace=False)
            if i not in chosen:
                chosen = np.append(chosen, i)
            # With i included
            with_data[idx, chosen] = explicand[0, chosen]
            if i in chosen:
                chosen = chosen[chosen != i]
            # Without i included
            without_data[idx, chosen] = explicand[0, chosen]
        with_vals = eval_model(with_data)
        without_vals = eval_model(without_data)
        X = np.concatenate((with_data, without_data))
        y = np.concatenate((with_vals, without_vals))
        learned_model = xgb.XGBRegressor(n_estimators=100, max_depth=4).fit(X, y)
        adjustment_with = learned_model.predict(with_data)
        adjustment_without = learned_model.predict(without_data)
        adjustment_with_centered = adjustment_with - np.mean(adjustment_with)
        adjustment_without_centered = adjustment_without - np.mean(adjustment_without)
        phi[0,i] = np.mean(with_vals - without_vals - adjustment_with_centered + adjustment_without_centered)
    return phi

def weighted_regression(baseline, explicand, model, num_samples, original_weight=True):
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64()) 
    all_s = np.array(list(range(1, num_features-1)))
    prob_s = 1 / ((num_features - all_s) * all_s)
    if not original_weight:
        prob_s = np.ones_like(prob_s)
    # Normalize
    prob_s = prob_s / prob_s.sum()
    sampled_s = gen.choice(all_s, num_samples-2, p=prob_s, replace=True)
    X = np.zeros((num_samples-2, num_features))
    fbaseline = eval_model(baseline)
    fexplicand = eval_model(explicand)
    for idx, s in enumerate(sampled_s):
        z = gen.choice(num_features, s, replace=False)
        X[idx, z] = 1
    diag_weights = 1 / (scipy.special.comb(num_features, sampled_s) * (num_features - sampled_s) * sampled_s)
    diag_weights = np.ones_like(diag_weights)
    W = np.diag(diag_weights)
    baseline_tiled = np.tile(baseline, (num_samples-2, 1))
    explicand_tiled = np.tile(explicand, (num_samples-2, 1))
    inputs = baseline_tiled * (1 - X) + explicand_tiled * X
    y = eval_model(inputs) - fbaseline
    # [ Q E^T] [ x ] = c
    # [ E 0 ] [ lambda ] = d
    Q = X.T @ W @ X
    c = X.T @ W @ y
    d = fexplicand - fbaseline
    E = np.ones(num_features)
    A = np.block([[Q, E.reshape(-1,1)], [E, 0]])
    b = np.append(c, d)
    phi = np.linalg.solve(A, b) 
    return phi[:-1]

def weighted_regression_leverage(baseline, explicand, model, num_samples):
    return weighted_regression(baseline, explicand, model, num_samples, original_weight=False)

def kernel_refined(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())

    all_s = np.array(list(range(1, num_features-1)))
    prob_s = 1 / ((num_features - all_s) * all_s)
    # Normalize
    prob_s = prob_s / prob_s.sum()
    sampled_s = gen.choice(all_s, (num_samples-2)//2, p=prob_s, replace=True)
    Z = np.zeros((num_samples-2, num_features))
    v0 = eval_model(baseline)
    v1 = eval_model(explicand)
    for idx, s in enumerate(sampled_s):
        z = gen.choice(num_features, s, replace=False)
        complement = np.setdiff1d(range(num_features), z)
        Z[2*idx, z] = 1
        Z[2*idx+1, complement] = 1
    inputs = baseline * (1 - Z) + explicand * Z
    vz = eval_model(inputs) - v0
    Z_1norm = Z.sum(axis=1)
    weights = 1 / (scipy.special.comb(num_features, Z_1norm) * (num_features - Z_1norm) * Z_1norm)
    b = vz.T @ (Z * weights[:, np.newaxis])
    correction = b.sum() - (v1 - v0) / num_features
    phi = num_features * b - correction
    return phi

def kernel_shap_base(baseline, explicand, model, num_samples):
    # https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley.py
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    Z = np.zeros((num_samples, num_features))
    v0 = eval_model(baseline)
    v1 = eval_model(explicand)
    for idx in range(num_samples):
        z = gen.choice(num_features, num_features // 2, replace=False)
        Z[idx, z] = 1
    inputs = baseline * (1 - Z) + explicand * Z
    vz = eval_model(inputs) - v0
    Z_1norm = Z.sum(axis=1)
    weights = 1 / (scipy.special.comb(num_features, Z_1norm) * (num_features - Z_1norm) * Z_1norm)
    b_hat = vz.T @ (Z)# * weights[:, np.newaxis])
    A_hat = Z.T @ (Z )#* weights[:, np.newaxis])
    ones = np.ones((num_features, 1))
    A_hat_inv = np.linalg.pinv(A_hat)
    scalar = (ones.T @ A_hat_inv @ b_hat - (v1 - v0)) / (ones.T @ A_hat_inv @ ones)
    phi = A_hat_inv @ (b_hat - scalar * ones)
    return phi


estimators = {
    'Permuation SHAP': permutation_shap,
    'Kernel SHAP': kernel_shap,
    'Sampling SHAP': shapley_sampling,
    'Tree SHAP': tree_shap,
    # Same samples of v(S) for all features
    'Recycled Sampling' : recycled_sampling,
    # Different samples of v(i|S) for each feature
    'Uniform Sampling' : uniform_sampling,
    # Adjust with prediction for v(i|S)
    'Uniform Sampling Adjusted' : uniform_sampling_adjusted,
    # Adjust with prediction for v(S cup i) and v(S)
    'Uniform Sampling Adjusted 2x' : uniform_sampling_adjusted2,
#    'Uniform Sampling Clever' : uniform_sampling_clever,
    'Uniform Sampling Sum' : uniform_sampling_sum,
    'Permutation Recycling' : permutation_recycling,
#    'Complementary Contribution' : complementary_contribution, # Slow
#    'Embedded Lattice' : embedded_lattice,
    'Weighted Regression Weighted Samples' : weighted_regression,
    'Weighted Regression Leverage Samples' : weighted_regression_leverage,
    'Refined Kernel SHAP' : kernel_refined,
    'Kernel SHAP Base' : kernel_shap_base
}

#for offset in [0,10,20,30,40,50,60,70,80,90,100]:
#    estimators[f'Uniform Sampling Offset {offset}'] = lambda baseline, explicand, model, num_samples, offset=offset : uniform_sampling_offset(baseline, explicand, model, num_samples, offset=offset)
