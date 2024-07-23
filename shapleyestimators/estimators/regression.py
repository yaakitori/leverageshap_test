import numpy as np
import scipy

# Compute b
# Compute A
# Compute v1-v0

def get_components(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False, weighted_problem=True):
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    Z = np.zeros((num_samples-2, num_features))
    v0 = eval_model(baseline)
    v1 = eval_model(explicand)
    
    # Select weights to sample each size
    valid_sizes = np.array(list(range(1, num_features)))
    if leverage_sampling:
        prob_s = np.ones(num_features-1)
    else:        
        prob_s = 1 / (valid_sizes * (num_features - valid_sizes))
    prob_s = prob_s / np.sum(prob_s)

    sampled_sizes = gen.choice(valid_sizes, num_samples-2, p=prob_s)
    for idx, s in enumerate(sampled_sizes):
        z = gen.choice(num_features, s, replace=False)
        if not paired_sampling:
            Z[idx, z] = 1
        else: # Add complement if paired sampling        
            complement = np.setdiff1d(np.arange(num_features), z)
            Z[2 * idx, z] = 1
            Z[2 * idx + 1, complement] = 1
            # Break half we through if paired sampling
            if idx >= num_samples // 2 - 2: break
    
    baseline_tiled = np.tile(baseline, (num_samples-2, 1))
    explicand_tiled = np.tile(explicand, (num_samples-2, 1))
    inputs = baseline_tiled * (1 - Z) + explicand_tiled * Z
    vz = eval_model(inputs)

    z_sizes = np.sum(Z, axis=1)
    weights = 1 / (z_sizes * (num_features - z_sizes) * scipy.special.binom(num_features, z_sizes))

    used_weights = weights if weighted_problem else np.ones_like(weights)
    
    A_hat = Z.T @ (Z * used_weights[:, np.newaxis]) 
    b_hat = (vz - v0) @ (Z * used_weights[:, np.newaxis])

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

def exact_regression(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=False, weighted_problem=True):
    components = get_components(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=leverage_sampling, weighted_problem=weighted_problem)
    A_hat = components['A_hat']
    b_hat = components['b_hat']
    delta = components['delta']
    num_features = baseline.shape[1]
    return (
        num_features * b_hat -
        (np.sum(b_hat) - delta / num_features)
    )

def exact_regression_leverage(baseline, explicand, model, num_samples, paired_sampling=True, weighted_problem=True):
    return exact_regression(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=True, weighted_problem=weighted_problem)

        
