# -*- coding: utf-8 -*-
# ============================================================
# 04_q3_tstr.py — 问题3：合成数据 + 用于预测 Gender 的 TSTR
# 复现 synthpop(CART) 的序列合成：
#   按访问顺序逐一变量拟合 CART（minbucket=5，与 synthpop 默认一致），
#   对每条合成记录走到叶节点后，从该叶内的真实观测取值中随机抽取
# 输出：保真度对照表、真实/合成数据 GLM 系数对照、TSTR AUC
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from plotstyle import setup

setup()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
rng = np.random.default_rng(7)

df_raw = pd.read_csv(os.path.join(OUT, "df_raw.csv"))

# ============================================================
# 划分：df_synth_input (80,000) 与 df_real_holdout (2,000)
# ============================================================
idx = rng.permutation(len(df_raw))
hold_idx = idx[:2000]
syn_idx = idx[2000:82000]
df_synth_input = df_raw.iloc[syn_idx].reset_index(drop=True)
df_real_holdout = df_raw.iloc[hold_idx].reset_index(drop=True)
print(f"df_synth_input: {len(df_synth_input)} 条; df_real_holdout: {len(df_real_holdout)} 条")


# ============================================================
# synthpop 式 CART 序列合成器
# ============================================================
def cart_synthesize(real, visit_seq, cat_vars, seed=7):
    """逐变量拟合 CART 并生成合成数据（叶内抽样，与 synthpop 'cart' 一致）。
    real: 全部为数值列；cat_vars 中的变量按分类树处理。"""
    r = np.random.default_rng(seed)
    syn = pd.DataFrame(index=range(len(real)))
    for var in visit_seq:
        preds = [p for p in visit_seq if p in syn.columns]  # 此前已合成的变量
        if not preds:
            # 第一个变量：直接从真实边际中带放回抽取
            syn[var] = r.choice(real[var].to_numpy(), size=len(real), replace=True)
            continue
        Xfit = real[preds].to_numpy()
        yfit = real[var].to_numpy()
        if var in cat_vars:
            tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)
        else:
            tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)
        tree.fit(Xfit, yfit)
        # 每条合成记录走到叶节点，从叶内真实取值中均匀抽取
        Xsyn = syn[preds].to_numpy()
        leaves_fit = tree.apply(Xfit)
        leaf_syn = tree.apply(Xsyn)
        # 预先按叶组织真实取值
        order = np.argsort(leaves_fit, kind="stable")
        leaves_sorted = leaves_fit[order]
        vals_sorted = yfit[order]
        leaf_values = {}
        for lf in np.unique(leaves_sorted):
            lo = np.searchsorted(leaves_sorted, lf, "left")
            hi = np.searchsorted(leaves_sorted, lf, "right")
            leaf_values[lf] = vals_sorted[lo:hi]
        draw = np.empty(len(real))
        for lf in np.unique(leaf_syn):
            m = leaf_syn == lf
            pool = leaf_values[lf]
            draw[m] = pool[r.integers(len(pool), size=m.sum())]
        syn[var] = draw
    return syn


VISIT = ["GenderMale", "Age", "Density", "Value", "Bonus", "HasClaim", "ClaimAmount"]
CAT = ["GenderMale", "HasClaim"]

# 原始连续变量（df_raw 只含分组后的列，需从原始 CSV 取回连续值）
src = pd.read_csv(os.path.join(BASE, "data", "pg15training_raw.csv"),
                  usecols=["Age", "Density", "Value", "Bonus", "ClaimNb", "ClaimTotal"])
src["HasClaim"] = (src["ClaimNb"] > 0).astype(int)
src["ClaimAmount"] = src["ClaimTotal"]
src = src.drop(columns=["ClaimNb", "ClaimTotal"]).reset_index(drop=True)

real_for_synth = pd.concat([
    pd.DataFrame({"GenderMale": (df_synth_input["Gender"] == "Male").astype(int).to_numpy()}),
    src.iloc[syn_idx].reset_index(drop=True),
], axis=1)

syn_df = cart_synthesize(real_for_synth, VISIT, CAT, seed=7)
syn_df["Gender"] = np.where(syn_df["GenderMale"] == 1, "Male", "Female")
real_for_synth["Gender"] = np.where(real_for_synth["GenderMale"] == 1, "Male", "Female")
# 合成数据的分组列（供后续使用，与 df_raw 同一套切点）
syn_df["AgeGroup"] = pd.cut(syn_df["Age"], [17, 25, 35, 45, 55, 65, 200],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
syn_df["DensityGroup"] = pd.cut(syn_df["Density"], [0, 50, 100, 200, np.inf],
                                labels=["Rural", "Suburban", "Urban", "Dense urban"])
syn_df["ValueGroup"] = pd.cut(syn_df["Value"], [0, 5000, 12000, 28000, 45000, np.inf],
                              labels=["Low", "Medium", "High", "Luxury", "Very high"])
syn_df["BonusGroup"] = pd.cut(syn_df["Bonus"], [-np.inf, -25, 25, 75, np.inf],
                              labels=["Low", "Medium", "High", "Very high"])
syn_df.to_csv(os.path.join(OUT, "syn_df.csv"), index=False)
print("syn_df 已生成:", syn_df.shape)

# ============================================================
# 保真度检查：汇总统计量（对照讲义表样式）
# ============================================================
def stats_row(d, name):
    claims = d[d["HasClaim"] == 1]
    return {
        "Statistic": name,
        "Mean age": round(d["Age"].mean(), 2),
        "SD age": round(d["Age"].std(ddof=1), 2),
        "Claim frequency": f"{d['HasClaim'].mean()*100:.1f}%",
        "Mean claim amount (given claim)": round(claims["ClaimAmount"].mean(), 0) if len(claims) else np.nan,
        "% Female": f"{(d['Gender']=='Female').mean()*100:.1f}%",
        "Median vehicle value": round(d["Value"].median(), 0),
        "Mean density": round(d["Density"].mean(), 1),
    }

fid = pd.DataFrame([
    stats_row(real_for_synth, "Real"),
    stats_row(syn_df, "Synthetic"),
])
print("\n===== 保真度检查：真实 vs 合成 =====")
print(fid.to_string(index=False))
fid.to_csv(os.path.join(OUT, "q3_fidelity.csv"), index=False)

# ---- 图：边际分布（Age 原尺度；Density、Value 按 log1p 尺度，右偏长尾）----
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
axes[0].hist(real_for_synth["Age"], bins=40, alpha=0.55, density=True,
             label="Real", color="#4C72B0")
axes[0].hist(syn_df["Age"], bins=40, alpha=0.55, density=True,
             label="Synthetic", color="#DD8452")
axes[0].set_title("Age"); axes[0].set_xlabel("岁"); axes[0].legend(fontsize=8)
for ax, col, nm in [(axes[1], "Density", "Density（log1p 尺度）"),
                    (axes[2], "Value", "Value（log1p 尺度）")]:
    ax.hist(np.log1p(real_for_synth[col]), bins=40, alpha=0.55, density=True,
            label="Real", color="#4C72B0")
    ax.hist(np.log1p(syn_df[col]), bins=40, alpha=0.55, density=True,
            label="Synthetic", color="#DD8452")
    ax.set_title(nm); ax.legend(fontsize=8)
fig.suptitle("边际分布：真实数据与合成数据的比较（CART 合成，n=80,000）", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_q3_marginals.png"), dpi=150, bbox_inches="tight")
print("图已保存: fig_q3_marginals.png")

# ============================================================
# 保真度检查：GLM 系数（HasClaim ~ Age + GenderMale + log1p(Density)
#                    + log1p(Value) + Bonus），对照讲义表
# ============================================================
def design(d):
    X = pd.DataFrame({
        "Age": d["Age"],
        "GenderMale": (d["Gender"] == "Male").astype(float),
        "log1p(Density)": np.log1p(d["Density"]),
        "log1p(Value)": np.log1p(d["Value"]),
        "Bonus": d["Bonus"],
    })
    return X

Xr, yr = design(real_for_synth), real_for_synth["HasClaim"].to_numpy()
Xs, ys = design(syn_df), syn_df["HasClaim"].to_numpy()

def fit_logit(X, y):
    m = LogisticRegression(C=np.inf, max_iter=2000)
    m.fit(X, y)
    return m

m_real, m_syn = fit_logit(Xr, yr), fit_logit(Xs, ys)
coef = pd.DataFrame({
    "Coefficient": Xr.columns,
    "Real data": np.round(m_real.coef_[0], 4),
    "Synthetic data": np.round(m_syn.coef_[0], 4),
})
coef["Difference"] = np.round(coef["Synthetic data"] - coef["Real data"], 4)
print("\n===== GLM 系数对照（HasClaim 频率模型，对照讲义）=====")
print(coef.to_string(index=False))
coef.to_csv(os.path.join(OUT, "q3_glm_coef.csv"), index=False)

# ============================================================
# 问题3：TSTR — 预测 Gender（Age, log1p(Density), log1p(Value), Bonus）
# ============================================================
def design_g(d):
    return pd.DataFrame({
        "Age": d["Age"],
        "log1p(Density)": np.log1p(d["Density"]),
        "log1p(Value)": np.log1p(d["Value"]),
        "Bonus": d["Bonus"],
    })

def auc_manual(y_true, scores):
    """讲义的 auc_manual：秩统计（Mann-Whitney）计算 AUC。"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列（平均秩）
    su = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    return (su[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

Xh_src = src.iloc[hold_idx].reset_index(drop=True)
df_holdout_full = pd.concat([df_real_holdout, Xh_src[["Density", "Value", "Bonus"]]], axis=1)
Xh = design_g(df_holdout_full)
yh = (df_real_holdout["Gender"] == "Male").astype(int).to_numpy()

mg_real = fit_logit(design_g(real_for_synth),
                    (real_for_synth["Gender"] == "Male").astype(int).to_numpy())
mg_syn = fit_logit(design_g(syn_df),
                   (syn_df["Gender"] == "Male").astype(int).to_numpy())

auc_real = auc_manual(yh, mg_real.predict_proba(Xh)[:, 1])
auc_syn = auc_manual(yh, mg_syn.predict_proba(Xh)[:, 1])
tstr = pd.DataFrame({
    "Model trained on": ["Real data", "Synthetic data"],
    "AUC on real holdout (predicting Gender)": [round(auc_real, 4), round(auc_syn, 4)],
})
print("\n===== TSTR：预测 Gender（2,000 条真实留出记录）=====")
print(tstr.to_string(index=False))
print(f"留出集中男性比例: {yh.mean()*100:.1f}%")
tstr.to_csv(os.path.join(OUT, "q3_tstr_gender.csv"), index=False)
