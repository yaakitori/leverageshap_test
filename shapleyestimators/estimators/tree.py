import shap

def tree_shap(baseline, explicand, model, num_samples):
    explainer = shap.TreeExplainer(model, baseline)
    shap_values = explainer.shap_values(explicand)
    return shap_values