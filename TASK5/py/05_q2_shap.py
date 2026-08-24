# ============================================================
# 05_q2_shap.py — 问题2：全局 SHAP + 单保单瀑布图
# a. 全局 SHAP：找出除 Age/Bonus 外影响最大的两个特征
# b. 测试集选一份非讲座示例保单，局部 SHAP 瀑布图
# c. 面向客户的解释（写入报告）
# ============================================================
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
sys.path.insert(0, os.path.join(BASE, "py"))
import plotstyle  # noqa

X_train = pd.read_parquet(os.path.join(OUT, "X_train.parquet"))
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))
y_test = pd.read_parquet(os.path.join(OUT, "y_test.parquet"))
with open(os.path.join(OUT, "freq_model_nolink.pkl"), "rb") as f:
    mdl = pickle.load(f)
bst = mdl["bst"]
preds = np.load(os.path.join(OUT, "freq_pred_nolink.npz"))
rate_pred = preds["rate_pred"]  # 全量测试集年化预测频率

# SHAP 子样本（讲座：子样本上计算）
X_s_orig = X_test.sample(n=3000, random_state=11)
orig_rows = X_s_orig.index.to_numpy()      # 原始行号（用于索引 freq_te）
X_s = X_s_orig.reset_index(drop=True)
e_s = y_test["Exposure"].iloc[orig_rows].to_numpy()

# TreeSHAP（对 XGBoost，用模型边际尺度即 log 尺度）
explainer = shap.TreeExplainer(bst, feature_perturbation="interventional",
                               data=X_s.iloc[:100].to_numpy() if False else None)
shap_values = explainer.shap_values(X_s)

# ---- a. 分组 SHAP 重要性（独热特征归组为原始定价变量） ----
groups = pd.read_csv(os.path.join(OUT, "feature_groups.csv"))
gmap = dict(zip(groups["feature"], groups["group"]))
feat_cols = list(X_s.columns)
col_groups = [gmap.get(c, c) for c in feat_cols]

df_grp = pd.DataFrame({"col": feat_cols, "grp": col_groups,
                       "mean_abs": np.abs(shap_values).mean(axis=0)})
grp_imp = df_grp.groupby("grp")["mean_abs"].sum().sort_values(ascending=False)
print("===== 分组 SHAP 重要性 =====")
print(grp_imp.round(5).to_string())

# 除 Age、Bonus 外影响最大的两个特征
top2 = grp_imp.drop(index=["Age", "Bonus"], errors="ignore").head(2)
print("\n除 Age/Bonus 外影响最大的两个特征:")
print(top2.round(5).to_string())

# ---- 图1：分组 SHAP 重要性条形图 ----
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(grp_imp.index[::-1], grp_imp.values[::-1], color="#4C72B0")
ax.set_xlabel("平均 |SHAP| 值（模型边际尺度）", fontsize=11)
ax.set_title("分组 SHAP 重要性：按定价变量汇总（Tweedie 频率模型）", fontsize=12)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q2a_shap_importance.png"), dpi=200)
plt.close(fig)
print("\n图已保存: q2a_shap_importance.png")

# ---- b. 单保单瀑布图（测试集、非讲座示例保单） ----
# 讲座示例保单: Age=51, Bonus=110, Density=236.4, 第90百分位预测纯保费
# 我们选一份"预测年化频率接近测试集第90百分位数"的保单
q90 = np.quantile(rate_pred, 0.90)
# 在 SHAP 子样本中定位：用其原始行号索引 rate_pred
freq_s = rate_pred[orig_rows]  # 子样本对应的预测频率
# 避开讲座示例（Age=51 且 Bonus=110 附近），选预测频率接近 q90 的
cand = np.argsort(np.abs(freq_s - q90))
sel = None
for i in cand:
    age = X_s.iloc[i]["Age"]
    bonus = X_s.iloc[i]["Bonus"]
    if abs(age - 51) > 3 and abs(bonus - 110) > 15:
        sel = i
        break
if sel is None:
    sel = int(cand[0])
sel_row_idx = orig_rows[sel]
print(f"\n所选保单: 原始行号 {sel_row_idx}, SHAP 子样本位置 {sel}")
print(f"Age={X_s.iloc[sel]['Age']}, Bonus={X_s.iloc[sel]['Bonus']}, "
      f"Density={X_s.iloc[sel]['Density']:.1f}, Exposure={e_s[sel]:.3f}")
print(f"预测年化频率={freq_s[sel]:.4f} (测试集第90分位={q90:.4f})")

# 瀑布图（对数边际尺度：log 预测年化频率）
base_val = explainer.expected_value
print(f"基准值 E[f(X)] = {base_val:.4f} (对数尺度), 预测 f(x) = {explainer.expected_value + shap_values[sel].sum():.4f}")
print(f"预测年化频率 = exp(预测边际) = {np.exp(explainer.expected_value + shap_values[sel].sum()):.4f} (模型直接预测: {freq_s[sel]:.4f})")

# 归组后的瀑布数据：连续变量保留原特征，独热变量归组求和
shap_sel = shap_values[sel]
x_sel = X_s.iloc[sel]
grp_order = []
grp_contrib = {}
for c, sv, xv in zip(feat_cols, shap_sel, x_sel.to_numpy()):
    g = gmap.get(c, c)
    grp_contrib.setdefault(g, 0.0)
    grp_contrib[g] += sv
# 只显示贡献显著的变量（按 |贡献| 排序，取前8）
contrib_items = sorted(grp_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
print("\n===== 该保单的各变量贡献（对数尺度）=====")
for g, v in contrib_items:
    print(f"  {g}: {v:+.4f}")

# 用 shap.force / waterfall 作图（shap 0.52 waterfall 需解释器+base值）
fig2 = plt.figure(figsize=(10, 5.5))
sv_sel = shap_values[sel]
shap.plots.waterfall(shap.Explanation(sv_sel, base_values=base_val,
                                       feature_names=feat_cols,
                                       data=X_s.iloc[[sel]].to_numpy()[0]),
                     max_display=12, show=False)
plt.tight_layout()
fig2.savefig(os.path.join(OUT, "q2b_shap_waterfall.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)
print("\n图已保存: q2b_shap_waterfall.png")

# 保存关键数值供报告使用
np.savez(os.path.join(OUT, "q2_shap.npz"),
         grp_names=np.array(grp_imp.index), grp_imp=grp_imp.values,
         base_val=base_val, sel_row=sel_row_idx, sel_pos=sel,
         sel_freq=freq_s[sel], q90=q90,
         sel_age=X_s.iloc[sel]["Age"], sel_bonus=X_s.iloc[sel]["Bonus"],
         sel_density=X_s.iloc[sel]["Density"],
         contrib_names=np.array([g for g, v in contrib_items]),
         contrib_vals=np.array([v for g, v in contrib_items]))
print("\nSHAP 分析完成")
