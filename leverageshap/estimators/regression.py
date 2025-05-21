import numpy as np
import scipy
import scipy.special
import math

# 「マスク行列 𝑍 の各行（どの特徴量を 1 にするか）」 を作る土台が combination_generator
def ith_combination(pool: range, r: int, index: int):
    # Function written by ChatGPT
    """
    pool の要素からちょうど r 個を 辞書順で選ぶとき、
    “index 番目” の組合せのみを直接計算する関数。全組合せを列挙せずに高速に取得できる。
    すべての組合せを列挙してから取り出すとメモリと時間がかかるので、計数（combinatorial counting）を使って一発で求める

    Args:
        pool (range): 要素を選ぶ元の集合
        r (int): 選択する要素の数（特徴量の部分集合のサイズ）
        index (int): 取得したい組合せのインデックス

    Returns:
        tuple: index 番目の組合せをタプルで返す
    """
    n = len(pool)
    combination = []
    elements_left = n
    k = r
    start = 0
    
    for i in range(r): # r個の特長量（要素）をサンプリング
        # Find the largest value for the first element in the combination
        # that allows completing the remaining k-1 elements
        for j in range(start, elements_left):
            count = math.comb(elements_left - j - 1, k - 1)
            if index < count:
                combination.append(pool[j])
                k -= 1
                start = j + 1
                break
            index -= count
    
    return tuple(combination)

## 各部分集合のサイズのサンプリングと そのサンプリングされたサイズの部分集合を生成 ##
def combination_generator(gen, n, s, num_samples):
    """
        Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
        1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
        2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
        Leverage SHAP では「特徴量 𝑛 個からサイズ s の部分集合を num_samples 個ランダムに取る」という操作を頻繁に行う。
        この関数はその部分集合をジェネレータとして順次返す
        Args:
            gen (np.random.Generator): numpy の random generator
            n (int): 特徴量数
            s (int): 部分集合のサイズ
            num_samples (int): 生成する部分集合の数
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)  # 0 ~ num_combos-1 の整数の中から、num_sample個の整数を重複なしでサンプリング
        for i in indices:
            yield ith_combination(range(n), s, i) # 特徴量のインデックスを要素に持つリストを出力
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)  # 要素を直接抽出（こちらは with replacement のような振る舞い）


class RegressionEstimator:
    def __init__(self, model, baseline, explicand, num_samples, paired_sampling=False, leverage_sampling=False, bernoulli_sampling=False):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand # そのままのデータ（1インスタンス）
        # Subtract 2 for the baseline and explicand and ensure num_samples is even
        self.num_samples = int((num_samples -2 ) // 2) * 2 # 必ず偶数にする
        self.paired_sampling = paired_sampling # 部分集合の補集合も合わせてサンプリング
        self.n = self.baseline.shape[1] # Number of features
        self.gen = np.random.Generator(np.random.PCG64())
        self.sample_weight = lambda s : 1 / (s * (self.n - s)) if not leverage_sampling else np.ones_like(s)
        self.reweight = lambda s : 1 / (self.sample_weight(s) * (s * (self.n - s)))
        self.kernel_weights = [] # 重みを格納するリスト、出力させたい
        self.sample = self.sample_with_replacement if not bernoulli_sampling else self.sample_without_replacement
        #self.used_indices = set()
    
    def add_one_sample(self, idx, indices, weight):
        """特徴量の組み合わせを作成し、重みも追加"""
        #indices = sorted(indices)
        #if tuple(indices) in self.used_indices: return
        #self.used_indices.add(tuple(indices))
        if not self.paired_sampling:
            self.SZ_binary[idx, indices] = 1
            self.kernel_weights.append(weight)
        else:
            indices_complement = np.array([i for i in range(self.n) if i not in indices]) # 補集合
            self.SZ_binary[2*idx, indices] = 1
            self.kernel_weights.append(weight)
            self.SZ_binary[2*idx+1, indices_complement] = 1 # 補集合
            self.kernel_weights.append(weight)

    
    def sample_with_replacement(self):
        self.SZ_binary = np.zeros((self.num_samples, self.n))
        valid_sizes = np.array(list(range(1, self.n)))
        prob_sizes = self.sample_weight(valid_sizes) # サイズに応じて重みづけ（leverage_sampling=Falseなら）
        prob_sizes = prob_sizes / np.sum(prob_sizes)
        num_sizes = self.num_samples if not self.paired_sampling else self.num_samples // 2 # paired_samplingは半分にする
        sampled_sizes = self.gen.choice(valid_sizes, num_sizes, p=prob_sizes) # 各サイズからサンプリングする数を決定
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False) # 1インスタンスからs個の特徴量をサンプリング（重複なし）
            # weight = Pr(sampling this set) * w(s)
            weight = 1 / (self.sample_weight(s) * s * (self.n - s))
            self.add_one_sample(idx, indices, weight=weight)
    
    def find_constant_for_bernoulli(self, max_C = 1e10):
        """Leverage SHAP の “全2**𝑛−2個ある部分集合 S を Bernoulli 抽選で取る”というサンプリング手法（without-replacement 版）
        Bernoulli Sampling のオーバーサンプリング定数 C を二分探索で決定する。
        目的：各部分集合サイズ s に対して p_s = min(1, 2*C*weight(s)/binom(self.n, s) ) という確率をかけて「取る/取らない」を決める。
        サンプリングしたときのサンプル数の期待値が m になるように C を探す。

        Args:
            max_C (_type_, optional): _description_. Defaults to 1e10.

        Returns:
            _type_: _description_
        """
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        C = 1 # Assume at least n - 1 samples
        m = min(self.num_samples, 2**self.n-2) # Maximum number of samples is 2^n -2
        def expected_samples(C):
            """定数 C のときに得られるサンプル数の期待値を返す。"""
            # scipy.special.binom(self.n, s) is the number of combinations of n choose s
            expected = [min(scipy.special.binom(self.n, s), 2* C * self.sample_weight(s)) for s in range(1, self.n)]
            #print(f'Expected samples: {np.sum(expected)}')
            #print(f'Constraint: {m}')
            #print(f'C: {C}')
            return np.sum(expected)
        # Efficiently find C with 二分探索
        L = 1
        R = scipy.special.binom(self.n, self.n // 2) * self.n ** 2
        while round(expected_samples(C)) != m: # 期待値がピッタリ m になるまでループ
            if expected_samples(C) < m:
                L = C  # 期待値が小さければ C を大きくする
            else:
                R = C # 期待値が大きければ C を小さくする
            C = (L + R) / 2
        self.C = round(C)
    
    def sample_without_replacement(self):
        """
        サンプリング without replacement（置換なし）を実行し、
        self.SZ_binary に二値マトリックス（どの特徴量を
        ON にしたか）と対応する重みを格納する。
        """
        self.find_constant_for_bernoulli() # オーバーサンプリング定数 C を決定する

        # 各部分集合サイズ s=1…n-1 について、実際に何個サンプリングするかを決定
        m_s_all = []
        for s in range(1, self.n):
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, C*sample_weight(s) / (n choose s))
            prob = min(1, 2*self.C * self.sample_weight(s) / scipy.special.binom(self.n, s))  # Bernoulli の成功確率
            try:
                m_s = self.gen.binomial(int(scipy.special.binom(self.n, s)), prob)  # 二項分布(ベルヌーイ試行をたくさん繰り返したときに、「成功」が何回起きるか)でサンプル数を得る
            except OverflowError:  # If the number of samples is too large, assume the number of samples is the expected number
                m_s = int(prob * scipy.special.binom(self.n, s)) # 期待値を使う
            if self.paired_sampling:
                if s == self.n // 2: # 中央サイズに到達したら、補集合を含めるとサンプリングが終了している
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        # n が偶数かつ中央サイズなら、重複してサンプリングしないよう半分に
                        m_s_all.append(m_s // 2)
                    else:
                        m_s_all.append(m_s)
                    break
            m_s_all.append(m_s)
        sampled_m = np.sum(m_s_all)
        num_rows = sampled_m if not self.paired_sampling else sampled_m * 2
        self.SZ_binary = np.zeros((num_rows, self.n))  # self.SZ_binary：GNNShapのmask_matrixに当たるので、出力したい
        idx = 0
        for s, m_s in enumerate(m_s_all):
            """
            各部分集合サイズsについて、事前に決めたサンプル数𝑚_s分だけ、combination_generator で「どの特徴量を選ぶか」の組合せを生成し、
            その組合せを二値マスク（self.SZ_binary の行）として格納。かつ、対応する重み weight を add_one_sample で設定
            """
            s += 1
            prob = min(1, 2*self.C * self.sample_weight(s) / scipy.special.binom(self.n, s)) # 論文中のベルヌーイ分布の式
            weight = 1 / (prob * scipy.special.binom(self.n, s) * (self.n - s) * s ) # Shapの計算に使う重み
            if self.paired_sampling and s == self.n // 2 and self.n % 2 == 0: # ペア付きかつ中央サイズの時
                # n-1 個から (s-1) 個を選ぶ組合せを生成し、最後に要素 n-1 を追加
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    # indices = 特徴量の組み合わせ
                    self.add_one_sample(idx, list(indices) + [self.n-1], weight = weight)
                    idx += 1
            else:
                # 通常の組合せ生成：0...n-1 から s 個選ぶ
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices), weight = weight)
                    idx += 1

    def compute(self):
        """
        SHAP値を計算する。main関数

        """
        # Sample
        self.sample()
        # A = Z P
        # y = v(z) - v0
        # b = y - Z1 (v1 - v0) / n
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        # Remove zero rows
        SZ_binary = self.SZ_binary[np.sum(self.SZ_binary, axis=1) != 0]
        # 基準値 v0 と対象値 v1 をモデルから予測
        # baseline: すべてのインスタンスで各特徴量での平均値の特徴量入力
        # explicand: 説明対象のインスタンスそのままの特徴量入力
        v0, v1 = self.model.predict(self.baseline), self.model.predict(self.explicand)
        # mask が1なら explicand の特徴量、0なら baseline の特徴量を選ぶ
        inputs = self.baseline * (1 - SZ_binary) + self.explicand * SZ_binary
        Sy = self.model.predict(inputs) - v0
        SZ_binary1 = np.sum(SZ_binary, axis=1)

        Sb = Sy - (v1 - v0) * SZ_binary1 / self.n

        # Projection matrix
        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        PZSSb = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ Sb
        PZSSZP = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ SZ_binary @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        self.phi = PZSSZP_inv_PZSSb + (v1 - v0) / self.n

        return self.phi

# TODO
def leverage_shap(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=True, leverage_sampling=True, bernoulli_sampling=True)
    return estimator.compute()

def leverage_shap_wo_bernoulli(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=True, leverage_sampling=True, bernoulli_sampling=False)
    return estimator.compute()

def leverage_shap_wo_bernoulli_paired(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=False, leverage_sampling=True, bernoulli_sampling=False)
    return estimator.compute()

def kernel_shap_paired(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=True, leverage_sampling=False, bernoulli_sampling=False)
    return estimator.compute() 

def kernel_shap(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=False, leverage_sampling=False, bernoulli_sampling=False)
    return estimator.compute()

def leverage_shap_wo_paired(baseline, explicand, model, num_samples):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=False, leverage_sampling=True, bernoulli_sampling=False)
    return estimator.compute()