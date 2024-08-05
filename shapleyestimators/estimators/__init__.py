import scipy.special
from .official_shap import *
from .regression import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'KernelSHAP': kernel_shap,
    'KernelSHAP Paired': kernel_shap_paired,
    #'KernelSHAP Optimized': kernel_shap_optimized,
    'Official KernelSHAP': official_kernel_shap,    
    'LeverageSHAP': kernel_shap_leverage,
    'LeverageSHAP Paired' : kernel_shap_leverage_paired,
    #'LeverageSHAP Optimized': kernel_shap_leverage_optimized,
    'Official Tree SHAP': official_tree_shap,
}
