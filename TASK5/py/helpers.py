# ============================================================
# 03_helpers.py — 复现讲座解释辅助函数
#   compute_pdp / compute_ale / 置换重要性 / group_feature_name
# 频率模型预测尺度：年化理赔频率 = E[ClaimNb] / Exposure
# ============================================================
import numpy as np
import pandas as pd
import xgboost as xgb

# 连续（未独热）特征：这些特征在设计中就是原始变量名
CONT_VARS = ["Age", "Bonus", "Density", "Poldur", "Value"]


def make_dmatrix(X, exposure):
    return xgb.DMatrix(X, base_margin=np.log(np.asarray(exposure, dtype=float)))


def predict_freq_annual(bst, X, exposure):
    """返回年化预测理赔频率 E[ClaimNb]/Exposure。"""
    dmat = xgb.DMatrix(X, base_margin=np.log(np.asarray(exposure, dtype=float)))
    return bst.predict(dmat) / np.asarray(exposure, dtype=float)


def group_feature_name(feature):
    """独热特征 -> 原始定价变量（讲座 group_feature_name）。"""
    return feature.split("_")[0]


def compute_pdp(bst, X, exposure, feature, grid, n_sample=None):
    """部分依赖图：固定 feature 为 grid 各取值，其余变量不变，平均年化频率。

    X        : 全量设计矩阵（DataFrame）
    exposure : 对应敞口
    feature  : 目标变量名（连续变量原始名 或 独热特征名）
    grid     : 网格取值
    n_sample : 子样本量（None=全部）
    """
    X = X.reset_index(drop=True)
    exposure = np.asarray(exposure)
    if n_sample is not None and n_sample < len(X):
        idx = np.random.default_rng(7).choice(len(X), n_sample, replace=False)
        Xw = X.iloc[idx].reset_index(drop=True)
        expw = exposure[idx]
    else:
        Xw, expw = X, exposure

    preds = []
    for g in grid:
        Xg = Xw.copy()
        Xg[feature] = g
        preds.append(predict_freq_annual(bst, Xg, expw).mean())
    return np.array(preds)


def compute_ale(bst, X, exposure, feature, grid, n_sample=None):
    """累积局部效应：区间内局部预测差异累积、居中（Apley & Zhu 2020）。

    返回 (grid_midpoints, ale_values)
    """
    X = X.reset_index(drop=True)
    exposure = np.asarray(exposure)
    if n_sample is not None and n_sample < len(X):
        idx = np.random.default_rng(7).choice(len(X), n_sample, replace=False)
        Xw = X.iloc[idx].reset_index(drop=True)
        expw = exposure[idx]
    else:
        Xw, expw = X, exposure

    xj = Xw[feature].to_numpy(dtype=float)
    k = len(grid) - 1  # 区间数
    ale = np.zeros(k)
    for j in range(k):
        lo, hi = grid[j], grid[j + 1]
        # 落入区间 (lo, hi] 的观测
        mask = (xj > lo) & (xj <= hi)
        if mask.sum() == 0:
            continue
        Xlo = Xw.copy()
        Xhi = Xw.copy()
        Xlo[feature] = lo
        Xhi[feature] = hi
        p_lo = predict_freq_annual(bst, Xlo.loc[mask], expw[mask])
        p_hi = predict_freq_annual(bst, Xhi.loc[mask], expw[mask])
        ale[j] = (p_hi - p_lo).mean()
    # 累积
    ale = np.cumsum(ale)
    # 居中：使 ALE 曲线相对平均预测解读
    weights = np.zeros(k)
    for j in range(k):
        mask = (xj > grid[j]) & (xj <= grid[j + 1])
        weights[j] = mask.sum()
    if weights.sum() > 0:
        ale = ale - np.average(ale, weights=weights)
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    return midpoints, ale


def permutation_importance(bst, X_test_df, y_te, e_te, feature_groups, n_repeats=5,
                           metric="poisson_deviance", n_sample=None):
    """置换重要性：打乱某定价变量全部编码特征，测量泊松偏差增幅（讲座）。

    返回 DataFrame: Variable, Importance, Std
    """
    X = X_test_df.reset_index(drop=True)
    y = np.asarray(y_te)
    e = np.asarray(e_te)
    if n_sample is not None and n_sample < len(X):
        idx = np.random.default_rng(0).choice(len(X), n_sample, replace=False)
        X, y, e = X.iloc[idx].reset_index(drop=True), y[idx], e[idx]

    def poisson_dev(y, mu):
        """2 * mean(y*log(y/mu) - (y-mu))，y=0 时首项约定为 0。"""
        mu = np.clip(mu, 1e-12, None)
        y = np.asarray(y, dtype=float)
        term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * np.mean(term1 - (y - mu))

    base = poisson_dev(y, predict_freq_annual(bst, X, e) * e)  # 偏差用 E[ClaimNb]=freq*E

    groups = sorted(feature_groups.keys())
    results = []
    rng = np.random.default_rng(2024)
    for g in groups:
        feats = feature_groups[g]
        per_rep = []
        for r in range(n_repeats):
            Xp = X.copy()
            for f in feats:
                Xp[f] = rng.permutation(Xp[f].to_numpy())
            dev_perm = poisson_dev(y, predict_freq_annual(bst, Xp, e) * e)
            per_rep.append(dev_perm - base)
        results.append((g, float(np.mean(per_rep)), float(np.std(per_rep))))
    res = pd.DataFrame(results, columns=["Variable", "Importance", "Std"])
    return res.sort_values("Importance", ascending=False).reset_index(drop=True)
