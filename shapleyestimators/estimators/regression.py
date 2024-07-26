import numpy as np
import scipy
import itertools

# Compute b
# Compute A
# Compute v1-v0

def get_components(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False, weighted_problem=True, use_determinisitic=False):
    num_samples -= 2 # Subtract 2 for the baseline and explicand
    num_samples = int((num_samples) // 2) * 2 # Make sure num_samples is even
    eval_model = lambda z : model.predict(explicand * z + baseline * (1 - z))
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    v0 = eval_model(np.zeros(num_features))
    v1 = eval_model(np.ones(num_features))
    
    # Select weights to sample each size
    valid_sizes = np.array(list(range(1, num_features)))
    if leverage_sampling:
        prob_s = np.ones(num_features-1)
    else:        
        prob_s = 1 / (valid_sizes * (num_features - valid_sizes))

    Z = np.zeros((num_samples, num_features))
    
    offset = 0
    if use_determinisitic:
        for size in range(1, num_features):
            if (num_samples) // (num_features -1) >= scipy.special.binom(num_features, size):
                for indices in itertools.combinations(range(num_features), size):
                    Z[offset, list(indices)] = 1
                    offset += 1
                prob_s[size-1] = 0

    if np.sum(prob_s) != 0:
        prob_s = prob_s / np.sum(prob_s)
        sampled_sizes = gen.choice(valid_sizes, num_samples-2-offset, p=prob_s)
        for idx, s in enumerate(sampled_sizes):
            indices = gen.choice(num_features, s, replace=False)
            if not paired_sampling:
                Z[offset + idx, indices] = 1
            if paired_sampling: # Add complement if paired sampling        
                indices_complement = np.array([i for i in range(num_features) if i not in indices])
                Z[offset + 2*idx, indices_complement] = 1
                Z[offset + 2*idx+1, indices] = 1
                # Break half way through if paired sampling
                if idx >= (num_samples - 2 - offset) // 2 -1: break
    Z1_norm = np.sum(Z, axis=1)

    # Remove zero rows
    Z = Z[Z1_norm != 0]
    Z1_norm = Z1_norm[Z1_norm != 0]    

    if leverage_sampling:
        inv_weights = Z1_norm * (num_features - Z1_norm)
    else:
        inv_weights = Z1_norm * (num_features - Z1_norm) * scipy.special.binom(num_features, Z1_norm)
    weights = 1 / inv_weights if weighted_problem else np.ones_like(inv_weights)

    inputs = baseline * (1 - Z) + explicand * Z
    vz = model.predict(inputs)

    A_hat = Z.T @ np.diag(weights) @ Z
    b_hat = Z.T @ np.diag(weights) @ (vz - v0)
    
    assert A_hat.shape == (num_features, num_features)
    assert b_hat.shape == (num_features,)

    return {
        'A_hat': A_hat,
        'b_hat': b_hat,
        'delta': v1 - v0,
    }

def kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=False, weighted_problem=True):
    components = get_components(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=leverage_sampling, weighted_problem=weighted_problem)
    A_hat = components['A_hat']
    b_hat = components['b_hat']
    delta = components['delta']
    A_hat_inv_ones = np.linalg.solve(A_hat, np.ones_like(b_hat))
    A_hat_inv_b_hat = np.linalg.solve(A_hat, b_hat)

    return (
        A_hat_inv_b_hat -
        A_hat_inv_ones * (
            np.sum(A_hat_inv_b_hat) - delta
        ) / np.sum(A_hat_inv_ones)
    )

def kernel_shap_leverage(baseline, explicand, model, num_samples, paired_sampling=True, weighted_problem=True):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=True, weighted_problem=weighted_problem)

def kernel_shap_unpaired(baseline, explicand, model, num_samples, weighted_problem=True):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False, weighted_problem=weighted_problem)

def kernel_shap_leverage_unpaired(baseline, explicand, model, num_samples, weighted_problem=True):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=True, weighted_problem=weighted_problem)


def weighted_regression(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=False, weighted_problem=True):
    components = get_components(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=leverage_sampling, weighted_problem=weighted_problem)
    A_hat = components['A_hat']
    b_hat = components['b_hat']
    delta = components['delta']
    # Solve
    # [ A_hat 1 ] [ x ] = [ b_hat ]
    # [ 1      0 ] [ lambda ] = [ delta ]
    matrix = np.block([[A_hat, np.ones((A_hat.shape[0], 1))], [np.ones((1, A_hat.shape[1])), 0]])
    
    delta = np.array([delta]).reshape((1,))
    vector = np.concatenate([b_hat, delta])
    return np.linalg.solve(matrix, vector)[:-1]

def weighted_regression_leverage(baseline, explicand, model, num_samples, paired_sampling=True, weighted_problem=True):
    return weighted_regression(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=True, weighted_problem=weighted_problem)
