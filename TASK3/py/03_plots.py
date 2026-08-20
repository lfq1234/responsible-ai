# ============================================================
# 03_plots.py — 生成作业报告所需图表
#   fig1: 基线公平性—准确性散点图（GLM + XGBoost 十点）
#   fig2: 问题1 lambda 扫描（DIR / RMSE 随 lambda 变化 + 公平性-准确性散点）
#   fig3: COMPAS roc_pivot theta 扫描（四大指标随 theta 变化）
#   fig4: 问题2 对比条形图（各去偏方案的 DIR）
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

MODEL_ORDER = ["M0: Full Model", "MU: Unawareness Model", "MDP: Demographic Parity",
               "MCDP: Conditional Demographic Parity", "MC: Controlling for the Protected Variable"]
COLORS = plt.cm.Set1(np.linspace(0, 1, 9))
MODEL_COLOR = {m: COLORS[i] for i, m in enumerate(MODEL_ORDER)}
METHOD_MARKER = {"GLM": "o", "XGBoost": "^"}

# ---------------- fig1: 基线公平性-准确性 ----------------
fa = pd.read_csv(os.path.join(OUT, "fairness_accuracy.csv"))
fig, ax = plt.subplots(figsize=(8, 5.5))
for (method, model), g in fa.groupby(["Method", "Model"]):
    ax.scatter(g["rmse"], g["disparate_impact_ratio"],
               color=MODEL_COLOR[model], marker=METHOD_MARKER[method],
               s=120, label=f"{model.split(':')[0]} ({method})", edgecolor="black", linewidth=0.5)
ax.axhline(0.8, color="grey", linestyle=":", linewidth=1)
ax.axhline(1.25, color="grey", linestyle=":", linewidth=1)
ax.axhline(1.0, color="grey", linestyle="-.", linewidth=1)
ax.set_xlabel("RMSE (Accuracy, lower is better)")
ax.set_ylabel("Disparate Impact Ratio (Fairness, closer to 1 is better)")
ax.set_title("Fairness–Accuracy Trade-off: GLM vs XGBoost Pricing Models")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.text(484, 1.30, "4/5th rule bounds [0.8, 1.25]", fontsize=8, color="grey")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_fairness_accuracy.png"), bbox_inches="tight")
plt.close()

# ---------------- fig2: lambda 扫描 ----------------
q1 = pd.read_csv(os.path.join(OUT, "q1_lambda_scan.csv"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(q1["lambda"], q1["dir"], marker="o", color="tab:red", linewidth=2)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Perfect fairness (DIR = 1)")
ax.set_xlabel("lambda (repair strength)")
ax.set_ylabel("Disparate Impact Ratio")
ax.set_title("(a) Fairness vs lambda")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(q1["lambda"], q1["rmse"], marker="s", color="tab:blue", linewidth=2)
ax.set_xlabel("lambda (repair strength)")
ax.set_ylabel("RMSE")
ax.set_title("(b) Accuracy vs lambda")
ax.grid(alpha=0.3)

plt.suptitle("Q1: MDP lambda scan (full-variable debiasing, GLM)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_lambda_scan.png"), bbox_inches="tight")
plt.close()

# fig2b: 公平性-准确性散点（标注 lambda）
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(q1["rmse"], q1["dir"], c=q1["lambda"], cmap="viridis", s=140, edgecolor="black")
for _, row in q1.iterrows():
    ax.annotate(f"lambda={row['lambda']:g}", (row["rmse"], row["dir"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("RMSE")
ax.set_ylabel("Disparate Impact Ratio")
ax.set_title("Q1: Fairness–Accuracy scatter across lambda")
cb = plt.colorbar(sc, ax=ax, label="lambda")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2b_lambda_fairness_acc.png"), bbox_inches="tight")
plt.close()

# ---------------- fig3: COMPAS theta 扫描 ----------------
scan = pd.read_csv(os.path.join(OUT, "compas_theta_scan.csv"))
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Parity = 1")
ax.plot(scan["theta"], scan["prop_parity"], marker="o", color="tab:red", linewidth=2, label="Demographic parity")
ax.plot(scan["theta"], scan["equal_odds"], marker="s", color="tab:orange", linewidth=2, label="Equalized odds (TPR gap)")
ax.plot(scan["theta"], scan["pred_rate_parity"], marker="^", color="tab:green", linewidth=2, label="Predictive rate parity")
ax.plot(scan["theta"], scan["accuracy"], marker="d", color="tab:blue", linewidth=2, label="Accuracy")
ax.axvspan(0.0, 0.20, color="grey", alpha=0.08)
ax.set_xlabel("theta (ROC pivot bandwidth)")
ax.set_ylabel("Metric value")
ax.set_title("Q5: roc_pivot post-processing on MU (theta extended to 0.30)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_compas_theta_scan.png"), bbox_inches="tight")
plt.close()

# ---------------- fig4: 各去偏方案 DIR 对比 ----------------
summary = json.load(open(os.path.join(OUT, "q1_q2_summary.json"), encoding="utf-8"))
labels = ["MU (no debias)", "MDP lambda=1\n(all vars)", "MCDP (lecture)\nInsurancescore only",
          "Q2: Density only", "Q1 lambda=0\n(≈MU)"]
dirs = [summary["reference_MU"]["dir"], q1.loc[q1["lambda"] == 1.0, "dir"].iloc[0],
        0.95695, summary["q2_density_only"]["dir"], q1.loc[q1["lambda"] == 0.0, "dir"].iloc[0]]
rmses = [summary["reference_MU"]["rmse"], q1.loc[q1["lambda"] == 1.0, "rmse"].iloc[0],
         492.99517, summary["q2_density_only"]["rmse"], q1.loc[q1["lambda"] == 0.0, "rmse"].iloc[0]]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(labels))
bars = ax.bar(x - 0.2, dirs, width=0.38, color="tab:red", label="Disparate Impact Ratio")
ax2 = ax.twinx()
ax2.plot(x + 0.2, rmses, marker="o", color="tab:blue", label="RMSE", linewidth=2)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("DIR (closer to 1 = fairer)", color="tab:red")
ax2.set_ylabel("RMSE", color="tab:blue")
ax.set_title("Q2: Which variable to debias? DIR comparison")
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="lower left", fontsize=8)
for i, (d, r) in enumerate(zip(dirs, rmses)):
    ax.annotate(f"{d:.3f}", (i - 0.2, d + 0.01), ha="center", fontsize=8, color="tab:red")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_debias_comparison.png"), bbox_inches="tight")
plt.close()

print("图表已生成:", os.listdir(OUT))
