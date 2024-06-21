import shap

def permutation_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)

    num_features = explicand.shape[1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, npermutations=num_permutations, silent=True)
    return shap_values