# -*- coding: utf-8 -*-
# ============================================================
# 02_q1_risk.py — 问题1：移除 AgeGroup 后评估重新识别风险，
#                并在四变量准标识符集合上应用 k=100 局部抑制
# 复现 sdcMicro 的方法：
#   个体风险 r_k = 1/f_k（文件按总体处理，无抽样权重）
#   全局风险 = sum(r_k)/n = 预期重新识别数 / n
#   localSuppression: 对等价类 < k 的记录，用最少个数的被抑制取值
#   （NA 按通配符匹配）使其等价类达到 k，importance 小者优先牺牲
# ============================================================
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import setup

setup()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")

df_raw = pd.read_csv(os.path.join(OUT, "df_raw.csv"))
QI5 = ["Gender", "AgeGroup", "DensityGroup", "ValueGroup", "BonusGroup"]
QI4 = ["Gender", "DensityGroup", "ValueGroup", "BonusGroup"]


# ------------------------------------------------------------
# 风险估计（对应 sdcMicro::createSdcObj + print(sdc, "risk")）
# ------------------------------------------------------------
def risk_summary(df, key_vars, label=""):
    """按总体语义计算个体风险 r_k=1/f_k 与全局风险。"""
    key = df[key_vars].astype(str)
    fk = key.groupby(key_vars, observed=False).transform("size").to_numpy()
    rk = 1.0 / fk
    n = len(df)
    res = {
        "label": label,
        "n": n,
        "global_risk_pct": 100 * rk.sum() / n,   # 预期重识别数/n
        "expected_reid": rk.sum(),
        "max_risk": rk.max(),
        "k1": int((fk == 1).sum()),
        "k3": int((fk <= 3).sum()),
        "k5": int((fk <= 5).sum()),
    }
    return res


def smallest_groups(df, key_vars, m=10):
    key = df[key_vars].astype(str)
    sz = key.groupby(key_vars, observed=False).size()
    return sz.sort_values().head(m).reset_index(name="n_records")


# ============================================================
# 基线：讲义的五变量结果（用于对照）
# ============================================================
base5 = risk_summary(df_raw, QI5, "5变量基线")
print("===== 五变量基线（对照讲义 0.77% / 765 / 6 / 40 / 102）=====")
print(pd.Series(base5).to_string())
print("\n最小的十个准标识符群组（五变量）:")
print(smallest_groups(df_raw, QI5).to_string(index=False))

# ============================================================
# 问题 1a：移除 AgeGroup 后的四变量风险
# ============================================================
base4 = risk_summary(df_raw, QI4, "4变量(移除AgeGroup)")
print("\n===== 问题1a：移除 AgeGroup 后 =====")
print(pd.Series(base4).to_string())
print("\n最小的十个准标识符群组（四变量）:")
print(smallest_groups(df_raw, QI4).to_string(index=False))

cmp = pd.DataFrame([
    ["全局重新识别风险", f"{base5['global_risk_pct']:.3f}%", f"{base4['global_risk_pct']:.3f}%"],
    ["预期重新识别数量", f"{base5['expected_reid']:.0f}", f"{base4['expected_reid']:.0f}"],
    ["最大个体风险", f"{base5['max_risk']:.2f}", f"{base4['max_risk']:.2f}"],
    ["k = 1 的记录数", base5["k1"], base4["k1"]],
    ["k <= 3 的记录数", base5["k3"], base4["k3"]],
    ["k <= 5 的记录数", base5["k5"], base4["k5"]],
], columns=["指标", "5变量基线", "4变量(移除AgeGroup)"])
print("\n", cmp.to_string(index=False))
cmp.to_csv(os.path.join(OUT, "q1a_compare.csv"), index=False)

# ============================================================
# 问题 1c：k=100 局部抑制（importance: Gender=1 ... Bonus=4）
# ============================================================
K = 100
IMPORTANCE = {"Gender": 1, "DensityGroup": 2, "ValueGroup": 3, "BonusGroup": 4}
# 候选抑制子集：先按"抑制个数"升序，同个数内按 importance 降序（先牺牲可舍弃变量）
vars_by_imp_desc = sorted(QI4, key=lambda v: -IMPORTANCE[v])  # Bonus, Value, Density, Gender
subsets = []
for size in range(1, 5):
    for combo in itertools.combinations(vars_by_imp_desc, size):
        subsets.append(list(combo))

vals = {v: df_raw[v].astype(str).to_numpy() for v in QI4}
n = len(df_raw)
suppressed = {v: np.zeros(n, dtype=bool) for v in QI4}

# 各变量子集上的等价类规模（通配符语义：NA 与任意取值匹配）
# 对每个"保留变量集合 S"，预先计算全体记录在 S 上的频数
from collections import defaultdict
size_lookup = {}
for r in range(0, 5):
    for S in itertools.combinations(QI4, r):
        if r == 0:
            size_lookup[S] = np.full(n, n)
        else:
            key = np.array(list(zip(*[vals[v] for v in S])))
            uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            size_lookup[S] = cnt[inv]

# 找出等价类 < K 的记录（四变量全保留时的规模）
fk_full = size_lookup[tuple(QI4)]
viol_idx = np.where(fk_full < K)[0]
print(f"\n===== 问题1c：k={K} 局部抑制 =====")
print(f"需要处理的记录数（等价类 < {K}）: {len(viol_idx)}")

# 贪心局部抑制：对每条违规记录，用最少个数的抑制使其通配符等价类 >= K
for i in viol_idx:
    for T in subsets:                      # T = 拟抑制的变量集合
        S = tuple(v for v in QI4 if v not in T)   # 保留的变量
        if size_lookup[S][i] >= K:
            for v in T:
                suppressed[v][i] = True
            break

sup_n = {v: int(suppressed[v].sum()) for v in QI4}
total_cells = sum(sup_n.values())
print("各准标识符抑制情况:")
sup_tbl = pd.DataFrame({
    "准标识符": QI4,
    "被抑制取值个数": [sup_n[v] for v in QI4],
    "抑制率": [f"{100*sup_n[v]/n:.2f}%" for v in QI4],
})
print(sup_tbl.to_string(index=False))
print("被抑制单元格总数:", total_cells)
sup_tbl.to_csv(os.path.join(OUT, "q1c_suppression.csv"), index=False)

# ------------------------------------------------------------
# 抑制后的数据与最终全局风险（NA 按通配符匹配）
# ------------------------------------------------------------
df_sup = df_raw.copy()
for v in QI4:
    df_sup.loc[suppressed[v], v] = np.nan

# 对每条记录，f = 与其在"非 NA 变量"上取值一致的记录数（候选记录的 NA
# 视为与任何取值匹配，与 sdcMicro 对缺失值的处理一致）：
#   f(r) = #{r' : 对 r 的每个非 NA 变量 v，r'[v] 为 NA 或等于 r[v]}
# 出现的模式集合只有少数几种（完全未抑制 / 仅 Bonus 被抑制），逐模式向量化计算。
sup_vals = {v: df_sup[v].astype(str).fillna("*").to_numpy() for v in QI4}
is_na = {v: sup_vals[v] == "*" for v in QI4}
nonna_pat = pd.Series([tuple(v for v in QI4 if not is_na[v][i]) for i in range(n)])
pat_codes, pat_uniq = pd.factorize(nonna_pat)
f_post = np.zeros(n)
for ci, S in enumerate(pat_uniq):
    idx_S = np.where(pat_codes == ci)[0]
    tup = np.array(list(zip(*[sup_vals[v][idx_S] for v in S])))
    for t in np.unique(tup, axis=0):
        tmap = dict(zip(S, t))
        agree = np.ones(n, dtype=bool)
        for v in S:
            agree &= is_na[v] | (sup_vals[v] == tmap[v])
        sel = (tup == t).all(axis=1)
        f_post[idx_S[sel]] = agree.sum()

rk_post = 1.0 / f_post
global_risk_post = 100 * rk_post.sum() / n
print(f"\n抑制后全局风险: {global_risk_post:.3f}%   "
      f"预期重新识别数: {rk_post.sum():.0f}   最大个体风险: {rk_post.max():.3f}")
with open(os.path.join(OUT, "q1c_result.txt"), "w", encoding="utf-8") as f:
    f.write(f"k={K}\n")
    f.write(f"被抑制单元格总数: {total_cells}\n")
    f.write(sup_tbl.to_string(index=False) + "\n")
    f.write(f"抑制后全局风险: {global_risk_post:.3f}%\n")
    f.write(f"抑制后预期重新识别数: {rk_post.sum():.0f}\n")
    f.write(f"抑制后最大个体风险: {rk_post.max():.3f}\n")

# ============================================================
# 图 1：k=100 抑制前后 BonusGroup 分布（含 NA 柱）
# ============================================================
order = ["Low", "Medium", "High", "Very high"]
before = df_raw["BonusGroup"].value_counts().reindex(order)
after = df_sup["BonusGroup"].value_counts(dropna=False).reindex(order + [np.nan]).fillna(0)
after.index = [('NA(被抑制)' if str(i)=='nan' else str(i)) for i in after.index]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(after.index))
ax.bar(x - 0.2, before.reindex(after.index).fillna(0).values, width=0.4,
       label="抑制前", color="#4C72B0")
ax.bar(x + 0.2, after.values, width=0.4, label="抑制后 (k=100)", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(after.index)
ax.set_ylabel("记录数")
ax.set_title(f"k={K} 局部抑制前后 BonusGroup 分布（承受抑制最多的变量）")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_q1_bonus_dist.png"), dpi=150)

# ============================================================
# 图 2：各准标识符抑制率
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 3.6))
rates = [100 * sup_n[v] / n for v in QI4]
ax.barh(QI4[::-1], rates[::-1], color="#55A868")
ax.set_xlabel("抑制率 (%)")
ax.set_title(f"k={K}、importance=Gender>...>BonusGroup 下各准标识符的抑制率")
for i, v in enumerate(rates[::-1]):
    ax.text(v + 0.1, i, f"{v:.2f}%", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_q1_suppression_rate.png"), dpi=150)
print("\n图已保存: fig_q1_bonus_dist.png, fig_q1_suppression_rate.png")
