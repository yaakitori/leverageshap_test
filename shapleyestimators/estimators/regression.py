import numpy as np
import scipy
import itertools

import scipy.special

# Compute b
# Compute A
# Compute v1-v0

def leverage_shap_new(baseline, explicand, model, num_samples, paired_sampling=False):
    num_samples -= 2 # Subtract 2 for the baseline and explicand
    num_samples = int((num_samples) // 2) * 2 # Make sure num_samples is even
    eval_model = lambda z : model.predict(explicand * z + baseline * (1 - z))
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    v0 = eval_model(np.zeros(num_features))
    v1 = eval_model(np.ones(num_features))

    # Projection matrix
    P = np.eye(num_features) - 1/num_features * np.ones((num_features, num_features))
    
    valid_sizes = np.array(list(range(1, num_features)))
    prob_sizes = np.ones_like(valid_sizes) / len(valid_sizes)

    Z = np.zeros((num_samples, num_features))
    
    sample_weights = np.zeros(num_samples)
    sampled_sizes = gen.choice(valid_sizes, num_samples, p=prob_sizes)
    for idx, s in enumerate(sampled_sizes):
        indices = gen.choice(num_features, s, replace=False)
        weighting = 1 / np.sqrt((scipy.special.comb(num_features, s) * (num_features -s) * s))
        sample_weighting = 1 / np.sqrt(scipy.special.comb(num_features, s) * (num_features - 1) * num_samples)
        #weighting = 1 / (scipy.special.comb(num_features, s))
        if not paired_sampling:
            Z[idx, indices] = weighting
            sample_weights[idx] = sample_weighting
        if paired_sampling: # Add complement if paired sampling
            indices_complement = np.array([i for i in range(num_features) if i not in indices])
            Z[2*idx, indices_complement] = weighting
            sample_weights[2*idx] = sample_weighting
            Z[2*idx+1, indices] = weighting
            sample_weights[2*idx+1] = sample_weighting
            # Break half way through if paired sampling
            if idx >= (num_samples - 2) // 2 -1: break
    
    # Convert to binary version of SZ
    Z_binary = (Z > 0).astype(int)
    inputs = baseline * (1 - Z_binary) + explicand * Z_binary
    y = model.predict(inputs) - v0

    Z1 = np.sum(Z, axis=1)
    b = y - (v1 - v0) * Z1 / num_features

    SA = np.diag(sample_weights) @ Z @ P

    Sb = sample_weights * b

    # (SA^T SA)^-1 SA^T Sb
    phi_perp = np.linalg.lstsq(SA, Sb, rcond=None)[0]

    phi = phi_perp + (v1 - v0) / num_features

    return phi

def leverage_shap_new_paired(baseline, explicand, model, num_samples):
    return leverage_shap_new(baseline, explicand, model, num_samples, paired_sampling=True)

def get_components(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False):
    num_samples -= 2 # Subtract 2 for the baseline and explicand
    num_samples = int((num_samples) // 2) * 2 # Make sure num_samples is even
    eval_model = lambda z : model.predict(explicand * z + baseline * (1 - z))
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    v0 = eval_model(np.zeros(num_features))
    v1 = eval_model(np.ones(num_features))
    
    # Real weights
    compute_weights = lambda s: 1 / (s * (num_features - s))
    # Select weights to sample each size
    distribution = lambda s: 1 / (s * (num_features - s)) if not leverage_sampling else np.ones_like(s)

    valid_sizes = np.array(list(range(1, num_features)))
    prob_s = distribution(valid_sizes)

    Z = np.zeros((num_samples, num_features))
    
    prob_s = prob_s / np.sum(prob_s)
    sampled_sizes = gen.choice(valid_sizes, num_samples-2, p=prob_s)
    for idx, s in enumerate(sampled_sizes):
        indices = gen.choice(num_features, s, replace=False)
        if not paired_sampling:
            Z[idx, indices] = 1
        if paired_sampling: # Add complement if paired sampling        
            indices_complement = np.array([i for i in range(num_features) if i not in indices])
            Z[2*idx, indices_complement] = 1
            Z[2*idx+1, indices] = 1
            # Break half way through if paired sampling
            if idx >= (num_samples - 2) // 2 -1: break
    Z1_norm = np.sum(Z, axis=1)

    # Remove zero rows
    Z = Z[Z1_norm != 0]
    Z1_norm = Z1_norm[Z1_norm != 0]

    reweighting = compute_weights(Z1_norm) / distribution(Z1_norm)

    inputs = baseline * (1 - Z) + explicand * Z
    vz = model.predict(inputs)

    ZTZ = Z.T @ np.diag(reweighting) @ Z
    ZTv = Z.T @ np.diag(reweighting) @ (vz - v0)
    
    assert ZTZ.shape == (num_features, num_features)
    assert ZTv.shape == (num_features,)

    return {
        'ZTZ': ZTZ,
        'ZTv': ZTv,
        'delta': v1 - v0,
    }

def kernel_shap(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False):
    components = get_components(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=leverage_sampling)
    ZTZ = components['ZTZ']
    ZTv = components['ZTv']
    delta = components['delta']
    ZTZ_inv_ones = np.linalg.solve(ZTZ, np.ones_like(ZTv))
    ZTZ_inv_ZTv = np.linalg.solve(ZTZ, ZTv)

    return (
        ZTZ_inv_ZTv -
        ZTZ_inv_ones * (
            np.sum(ZTZ_inv_ZTv) - delta
        ) / np.sum(ZTZ_inv_ones)
    )

def kernel_shap_paired(baseline, explicand, model, num_samples):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=False)

def kernel_shap_leverage(baseline, explicand, model, num_samples, paired_sampling=False):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=True)

def kernel_shap_leverage_paired(baseline, explicand, model, num_samples):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=True)
