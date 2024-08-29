import scipy.special
from .official_shap import *
from .regression import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'KernelSHAP': kernel_shap,
    'KernelSHAP Paired': kernel_shap_paired,
    'Official KernelSHAP': official_kernel_shap,    
    'LeverageSHAP': kernel_shap_leverage,
    'LeverageSHAP Paired' : kernel_shap_leverage_paired,
    'Official Tree SHAP': official_tree_shap,
    'LeverageSHAP New': leverage_shap_new,
    'LeverageSHAP New Paired': leverage_shap_new_paired,
}
