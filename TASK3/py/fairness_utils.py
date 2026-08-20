# ============================================================
# fairness_utils.py
# 复刻讲义使用的 R 包核心算法（严格对照 CRAN 源码）：
#   1. disparate_impact_remover  (fairmodels 包, Feldman et al. 2015)
#   2. roc_pivot                  (fairmodels 包, Kamiran et al. 2012)
#   3. 分类公平性指标             (fairness 包: prop_parity / equal_odds / pred_rate_parity)
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Disparate Impact Remover（几何修复法，保留秩）
#    fairmodels::disparate_impact_remover() 的 Python 复刻
# ------------------------------------------------------------

def _ecdf_values(vals: np.ndarray) -> np.ndarray:
    """R 的 ecdf(v)(v)：对每个观测返回 #(<= v)/n（右连续阶梯函数）。
    向量化：searchsorted 到排序数组右边界，得到 <= 该值的个数。"""
    vals = np.asarray(vals, dtype=float)
    sorted_vals = np.sort(vals)
    return np.searchsorted(sorted_vals, vals, side="right") / len(vals)


def _bucketize(num_buckets: int, x: np.ndarray) -> np.ndarray:
    """复刻 fairmodels:::bucketize()：把 CDF*100 切分到 num_buckets 个桶，
    返回 0-based 桶索引（R 中 factor levels 按区间从小到大排序，对应索引一致）。"""
    bin_borders = 2 * (num_buckets - 1)
    boundary = 100.0 / bin_borders
    y = np.cumsum(np.repeat(boundary, bin_borders))
    # R: seq(1, 2*(num_buckets-1), 2) 是 1-based；Python 0-based 取索引 0,2,4,...
    bins = y[np.arange(0, 2 * (num_buckets - 1), 2)]
    bins = np.concatenate([[0.0], bins, [100.0]])
    # cut(x*100, bins, include.lowest=TRUE)：第一个区间左闭，其余左开右闭
    idx = np.searchsorted(bins, x * 100.0, side="right") - 1
    idx = np.clip(idx, 0, num_buckets - 1)
    return idx


def disparate_impact_remover(df: pd.DataFrame, protected, features_to_transform,
                             lambda_: float = 1.0, rng: np.random.Generator = None):
    """调整 features_to_transform 中每一列的分布，使其不再依赖 protected 分组。
    lambda=0 几乎不变；lambda=1 完全修复（各组分位数映射到组间中位数）。
    返回修复后的 DataFrame 副本。"""
    if rng is None:
        rng = np.random.default_rng(14)
    result = df.copy()
    protected = pd.Series(protected).astype("category")
    levels = list(protected.cat.categories)
    # 注意：R 中 protected 需为 factor；此处 category 顺序即 levels 顺序

    for feature in features_to_transform:
        vals_by_group = {g: protected[protected == g].index.to_numpy() for g in levels}
        # num_buckets = min(各子组该特征 unique 值个数)，上限 101
        num_buckets = min(
            df.loc[vals_by_group[g], feature].nunique() for g in levels
        )
        num_buckets = min(num_buckets, 101)

        probs = np.linspace(0, 1, num_buckets)  # R: seq(0, 1, length.out = num_buckets)
        quantiles = {}
        for g in levels:
            Y = df.loc[vals_by_group[g], feature].to_numpy(dtype=float)
            quantiles[g] = np.quantile(Y, probs)  # R 默认 type=7 == numpy linear

        # inversed_Fa：各子组分位数的逐桶中位数（最小化 earth mover distance）
        inversed_Fa = np.median(np.vstack([quantiles[g] for g in levels]), axis=0)

        for g in levels:
            Y = df.loc[vals_by_group[g], feature].to_numpy(dtype=float)
            inversed_Fx = np.quantile(Y, probs)
            inversed_Fa_fixed = (1 - lambda_) * inversed_Fx + lambda_ * inversed_Fa

            Y_cdf = _ecdf_values(Y)
            bucket_idx = _bucketize(num_buckets, Y_cdf)
            Y_repaired = inversed_Fa_fixed[bucket_idx]
            result.loc[vals_by_group[g], feature] = Y_repaired
    return result


# ------------------------------------------------------------
# 2. roc_pivot（拒绝选项修正, Kamiran et al. 2012）
#    fairmodels::roc_pivot() 的 Python 复刻（纯预测修正版本）
#    假定：probs 越高 = 越有利；privileged = 优势群体
# ------------------------------------------------------------

def roc_pivot(probs: np.ndarray, protected, privileged, cutoff: float = 0.5,
              theta: float = 0.1) -> np.ndarray:
    """对落在 (cutoff-theta, cutoff+theta) 边界区间的预测做镜像翻转：
       - 优势群体：原本略高于 cutoff（有利）→ 翻转为不利
       - 非优势群体：原本略低于 cutoff（不利）→ 翻转为有利
    区间之外的预测保持不变。"""
    probs = np.asarray(probs, dtype=float).copy()
    protected = np.asarray(protected)
    is_close = np.abs(probs - cutoff) < theta
    is_privileged = protected == privileged
    is_favourable = probs > cutoff
    # 镜像：p -> cutoff - (p - cutoff) = 2*cutoff - p
    flip = is_close & is_privileged & is_favourable
    probs[flip] = cutoff - (probs[flip] - cutoff)
    flip = is_close & (~is_privileged) & (~is_favourable)
    probs[flip] = cutoff + (cutoff - probs[flip])
    probs = np.clip(probs, 0.0, 1.0)
    return probs


# ------------------------------------------------------------
# 3. 分类公平性指标（fairness 包同款定义）
#    positive = 正类（此处=再犯 yes）；base_group = 基准组（此处=白人）
#    所有比率 = 非裔 / 白人（与讲义表格一致）
# ------------------------------------------------------------

def binary_metrics_by_group(y_true, y_pred, group, base_group=None,
                            positive="yes"):
    """按组计算混淆矩阵指标与比率。返回 dict。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    groups = pd.Categorical(group).categories.tolist() if base_group is None \
        else [base_group] + [g for g in pd.Categorical(group).categories if g != base_group]
    base = base_group if base_group is not None else groups[0]

    def cm_stats(mask):
        yt, yp = y_true[mask], y_pred[mask]
        tp = np.sum((yp == positive) & (yt == positive))
        fp = np.sum((yp == positive) & (yt != positive))
        fn = np.sum((yp != positive) & (yt == positive))
        tn = np.sum((yp != positive) & (yt != positive))
        n = len(yt)
        acc = (tp + tn) / n if n else np.nan
        tpr = tp / (tp + fn) if (tp + fn) else np.nan   # sensitivity
        fpr = fp / (fp + tn) if (fp + tn) else np.nan   # 1 - specificity
        fnr = fn / (fn + tp) if (fn + tp) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        flagged = (yp == positive).mean()                # 被标记为正类的比例
        return dict(acc=acc, tpr=tpr, fpr=fpr, fnr=fnr, precision=precision,
                    flagged=flagged)

    stats = {g: cm_stats(group == g) for g in groups}
    base_stats = stats[base]

    def ratio(g, key):
        b = base_stats[key]
        return stats[g][key] / b if b and not np.isnan(b) else np.nan

    out = {"stats": stats}
    for g in groups:
        if g == base:
            continue
        # 人口统计学均等性（正向分类比例之比）——prop_parity
        out[f"prop_parity_{g}_vs_{base}"] = ratio(g, "flagged")
        # 机会均等/TPR 差距（非裔 TPR / 白人 TPR）——equal_odds
        out[f"equal_odds_{g}_vs_{base}"] = ratio(g, "tpr")
        # 预测率均等性（精确率之比）——pred_rate_parity
        out[f"pred_rate_parity_{g}_vs_{base}"] = ratio(g, "precision")
    return out


def compas_parity_report(y_true, y_pred, ethnicity, base="Caucasian", positive="yes"):
    """返回与讲义表格一致的指标行：准确率、被标记高风险比例、FPR、FNR、精确率、
    以及三大准则比率（非裔/白人）。"""
    r = binary_metrics_by_group(y_true, y_pred, ethnicity, base_group=base, positive=positive)
    s = r["stats"]
    groups = [g for g in s if g != base]
    line = {
        "accuracy": (s[base]["acc"] * len(np.asarray(ethnicity)[np.asarray(ethnicity) == base])
                     + sum(s[g]["acc"] * len(np.asarray(ethnicity)[np.asarray(ethnicity) == g]) for g in groups)) /
                    len(ethnicity),
        "flagged_white": s[base]["flagged"],
        "fpr_white": s[base]["fpr"],
        "fnr_white": s[base]["fnr"],
        "precision_white": s[base]["precision"],
    }
    for g in groups:
        line[f"flagged_{g}"] = s[g]["flagged"]
        line[f"fpr_{g}"] = s[g]["fpr"]
        line[f"fnr_{g}"] = s[g]["fnr"]
        line[f"precision_{g}"] = s[g]["precision"]
        line[f"prop_parity"] = r[f"prop_parity_{g}_vs_{base}"]
        line[f"equal_odds"] = r[f"equal_odds_{g}_vs_{base}"]
        line[f"pred_rate_parity"] = r[f"pred_rate_parity_{g}_vs_{base}"]
    return line
