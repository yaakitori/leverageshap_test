import shap

def shapley_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explainer = shap.SamplingExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True)
    return shap_values