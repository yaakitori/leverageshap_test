from .kernel import kernel_shap
from .sampling import shapley_sampling
from .tree import tree_shap

import numpy as np
import xgboost as xgb

#def regression_adjustment(baseline, explicand, model, num_samples):
#    num_features = baseline.shape[1]
#    gen = np.random.Generator(np.random.PCG64())
#    eval_model = lambda X : model.predict(X)
#    num_samples //= 2
#    # Learn a model to explain the predictions of the original model
#    synth_data = np.tile(baseline, (num_samples, 1))
#    for idx in range(num_samples):
#        # randomly choose a subset of features to mask
#        sample_size = gen.integers(1, num_features, endpoint=True)
#        chosen = gen.choice(range(num_features), sample_size, replace=False)
#        # create masked input as combination of baseline and explicand
#        synth_data[idx, chosen] = explicand[0, chosen]
#
#    predictions = eval_model(synth_data)
#    shap_model = xgb.XGBRegressor(n_estimators=100, max_depth=4).fit(synth_data, predictions)
#    chosens = np.zeros((num_samples, num_features))
#
#    # Estimate shapley values for the explicand with regression adjustment
#    synth_data = np.tile(baseline, (num_samples, 1))
#    for idx in range(num_samples):
#        sample_size = gen.integers(1, num_features, endpoint=True)
#        chosen = gen.choice(range(num_features), sample_size, replace=False)
#        chosens[idx, chosen] = 1
#        synth_data[idx, chosen] = explicand[0, chosen]
#
#    predictions = eval_model(synth_data)
#    adjustment = shap_model.predict(synth_data)
#    adjustment_centered = adjustment - np.mean(adjustment)
#
#    phi = np.zeros((2,num_features))
#    num_values = np.zeros((2,num_features))
#    for idx in range(num_samples):
#        val = predictions[idx] - adjustment_centered[idx]
#        for i in range(num_features):
#            sign = 1 if chosens[idx,i] == 1 else 0
#            phi[sign,i] += val
#            num_values[sign,i] += 1
#
#    phi /= num_values
#    phi = phi[1] - phi[0]
#    return phi

def recycled_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    synth_data = np.tile(baseline, (num_samples, 1))
    chosens = np.zeros((num_samples, num_features))
    for idx in range(num_samples):
        sample_size = gen.integers(1, num_features, endpoint=True)
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
            sample_size = gen.integers(1, num_features, endpoint=True)
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
            sample_size = gen.integers(1, num_features, endpoint=True)
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
            sample_size = gen.integers(1, num_features, endpoint=True)
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

estimators = {
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
    'Uniform Sampling Adjusted 2x' : uniform_sampling_adjusted2
}