import shap

def official_kernel_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)

    explainer = shap.KernelExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True)
    return shap_values

def official_permutation_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)

    num_features = explicand.shape[1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, npermutations=num_permutations, silent=True)
    return shap_values

def official_shapley_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explainer = shap.SamplingExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True)
    return shap_values

def official_tree_shap(baseline, explicand, model, num_samples):
    explainer = shap.TreeExplainer(model, baseline)
    shap_values = explainer.shap_values(explicand)
    return shap_values