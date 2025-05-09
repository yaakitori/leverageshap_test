import scipy.special
import shap
import numpy as np
import scipy
import itertools
import pandas as pd

def synthetic(num_features=15):
    binary = np.zeros((2**num_features-2, num_features))
    idx = 0
    for s in range(1, num_features):
        for indices in itertools.combinations(range(num_features), s):
            binary[idx, list(indices)] = 1
            idx += 1
    num_ones = np.sum(binary, axis=1)
    inv_weights = num_ones * (num_features - num_ones) * scipy.special.binom(num_features, num_ones)
    weights = 1 / inv_weights 
    Z = binary * weights[:, np.newaxis] # each row is w(||z||_1) z^T
    P = np.eye(num_features) - 1/num_features # projection matrix to remove all ones component
    A = Z @ P # each row is w(||z||_1) z^T P
    xstar = np.random.randn(num_features)
    ystar = A @ xstar
    weight_prob = weights / np.sum(weights)
    leverage = 1 / scipy.special.binom(num_features, num_ones)
    leverage_prob = leverage / np.sum(leverage)
    leverage_smaller = leverage_prob < weight_prob
    # Add noise
    noise = np.random.randn(2**num_features-2) * leverage_smaller
    # Convert to pandas dataframe
    X = pd.DataFrame(binary, columns=[f'Feature {i}' for i in range(num_features)])
    y = pd.Series(ystar + noise, name='Target')
    return X, y

dataset_loaders = {
    'Adult' : shap.datasets.adult,
    'California' : shap.datasets.california,
    'Communities' : shap.datasets.communitiesandcrime,
    'Correlated' : shap.datasets.corrgroups60,
    'Diabetes' : shap.datasets.diabetes,
    'Independent' : shap.datasets.independentlinear60,
    'IRIS' : shap.datasets.iris,
    'NHANES' : shap.datasets.nhanesi,
    'Synthetic' : synthetic,
}

def load_dataset(dataset_name):
    X, y = dataset_loaders[dataset_name]()
    # Remove nan values
    X = X.fillna(X.mean())
    return X, y # データセットの特徴量、正解ラベル

# X:データセットの特徴量(load_datasetで取得したX)
def load_input(X, seed=None, is_synthetic=False):
    if is_synthetic:
        baseline = np.zeros((1, X.shape[1])) # すべての特徴量が0（存在しない）のベースライン
        explicand = np.ones((1, X.shape[1])) # すべての特徴量が1（存在する存在する）の説明対象
        return baseline, explicand
    if seed is not None:
        np.random.seed(seed)
    baseline = X.mean().values.reshape(1, -1)  # 各特徴量の平均値をベースラインとして使用。.reshape(1, -1)で1行n列 の2次元配列に整形
    explicand_idx = np.random.choice(X.shape[0]) # 説明対象のインデックスをランダムに選択
    explicand = X.iloc[explicand_idx].values.reshape(1, -1) # 説明対象の行の特徴量を取得
    for i in range(explicand.shape[1]): # 説明対象のインスタンスの各特徴量について
        # baseline と explicandが各特徴量ごとに必ず異なる値を持つよう保証する
        # SHAP 等の説明手法では、ベースラインとの差分を計算するため、ある特徴量でベースラインと同値だと説明が得られない（差分が 0）というケースを避けたい、という意図がある
        while baseline[0, i] == explicand[0, i]:
            explicand_idx = np.random.choice(X.shape[0])
            explicand[0,i] = X.iloc[explicand_idx, i]
    return baseline, explicand