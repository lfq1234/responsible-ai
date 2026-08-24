# ============================================================
# 04_q1_pdp_ale.py — 问题1：新变量 Density 的 PDP 与 ALE
# 选择讲座未详析的变量：Density（人口密度）
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
from helpers import compute_pdp, compute_ale  # noqa
import plotstyle  # noqa

# 加载
X_train = pd.read_parquet(os.path.join(OUT, "X_train.parquet"))
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))
y_test = pd.read_parquet(os.path.join(OUT, "y_test.parquet"))
with open(os.path.join(OUT, "freq_model.pkl"), "rb") as f:
    mdl = pickle.load(f)
bst = mdl["bst"]

# 子样本（讲座：解释方法在子样本上计算以降低耗时）
X_s = X_test.sample(n=5000, random_state=5).reset_index(drop=True)
e_s = y_test["Exposure"].iloc[X_s.index].to_numpy()

# Density 取值范围与网格（分位数网格，覆盖观测范围）
dens = X_test["Density"].to_numpy()
grid = np.quantile(dens, np.linspace(0.01, 0.99, 51))
print("Density 范围:", dens.min(), "-", dens.max())
print("网格:", grid[:5], "...", grid[-5:])

pdp = compute_pdp(bst, X_s, e_s, "Density", grid)
mid, ale = compute_ale(bst, X_s, e_s, "Density", grid)

# ---- 图：PDP vs ALE ----
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(grid, pdp, label="PDP", color="#1f77b4", lw=2.2)
ax.plot(mid, ale, label="ALE", color="#d62728", lw=2.2, ls="--")
ax.set_xlabel("Density (人口密度, 居民/km²)", fontsize=11)
ax.set_ylabel("预测年化理赔频率", fontsize=11)
ax.set_title("Density 的主效应：PDP 与 ALE 对比（XGBoost 频率模型）", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q1_pdp_ale_density.png"), dpi=200)
plt.close(fig)
print("图已保存: q1_pdp_ale_density.png")

# ---- 输出表格：密度分箱支撑量（解释分歧用） ----
df_d = pd.DataFrame({"Density": dens})
bins = pd.cut(df_d["Density"], bins=grid)
counts = bins.value_counts().sort_index()
print("\n===== 各密度区间支撑量（测试集 30000 行）=====")
print(counts.head(8).to_string())
print("...")
print(counts.tail(8).to_string())

# 保存 PDP/ALE 数值
np.savez(os.path.join(OUT, "q1_pdp_ale.npz"), grid=grid, pdp=pdp, mid=mid, ale=ale,
         dens=dens, bin_counts=counts.to_numpy())
print("\nPDP 最大值:", pdp.max().round(4), "最小值:", pdp.min().round(4))
print("ALE 最大值:", ale.max().round(4), "最小值:", ale.min().round(4))
print("Density 分位数: ", np.quantile(dens, [0.5, 0.9, 0.95, 0.99]).round(1))
