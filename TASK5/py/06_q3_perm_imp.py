# ============================================================
# 06_q3_perm_imp.py — 置换重要性（问题 3a 实证依据）
# ============================================================
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
sys.path.insert(0, os.path.join(BASE, "py"))
import plotstyle  # noqa
from helpers import permutation_importance  # noqa

X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))
y_test = pd.read_parquet(os.path.join(OUT, "y_test.parquet"))
groups = pd.read_csv(os.path.join(OUT, "feature_groups.csv"))
with open(os.path.join(OUT, "freq_model.pkl"), "rb") as f:  # 有 offset 模型，基线稳定
    bst = pickle.load(f)["bst"]

# 归组：每个原始变量对应的所有独热特征
from collections import defaultdict
feature_groups = defaultdict(list)
for _, row in groups.iterrows():
    feature_groups[row["group"]].append(row["feature"])

# 置换重要性（子样本 10000 行，重复 5 次）—— 关键：用 sample 索引对齐 y/e
sub = X_test.sample(n=10000, random_state=3)
sub_idx = sub.index.to_numpy()
pi = permutation_importance(bst, sub,
                            y_test["ClaimNb"].iloc[sub_idx].to_numpy(),
                            y_test["Exposure"].iloc[sub_idx].to_numpy(),
                            dict(feature_groups), n_repeats=5, n_sample=None)
print("===== 置换重要性（Poisson 偏差增幅，5 次重复均值±std）=====")
print(pi.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# 图
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(pi["Variable"][::-1], pi["Importance"][::-1],
        xerr=pi["Std"][::-1], color="#4C72B0", alpha=0.8)
ax.set_xlabel("置换重要性：泊松偏差增幅", fontsize=11)
ax.set_title("置换重要性（XGBoost 频率模型，测试集 10000 行 × 5 重复）", fontsize=12)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q3_permutation_importance.png"), dpi=200)
plt.close(fig)
print("图已保存: q3_permutation_importance.png")

pi.to_csv(os.path.join(OUT, "q3_permutation_importance.csv"), index=False)
print("CSV 已保存: q3_permutation_importance.csv")
