import scipy.special
from .official_shap import *
from .regression import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'KernelSHAP': kernel_shap_unpaired,
    'KernelSHAP Paired': kernel_shap,
    'Official KernelSHAP': official_kernel_shap,
    'LeverageSHAP': kernel_shap_leverage_unpaired,
    'LeverageSHAP Paired' : kernel_shap_leverage,
    'Official Tree SHAP': official_tree_shap,
}
