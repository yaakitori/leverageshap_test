import scipy.special
from .official_shap import *
from .regression import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'Kernel SHAP': kernel_shap,
    'Official Kernel SHAP': official_kernel_shap,
    'Leverage SHAP': leverage_shap,
    'Kernel SHAP Paired': kernel_shap_paired,
    'Leverage SHAP wo Paired': leverage_shap_wo_paired,
    'Leverage SHAP wo Bernoulli, Paired': leverage_shap_wo_bernoulli_paired,
    'Official Tree SHAP': official_tree_shap,
}
