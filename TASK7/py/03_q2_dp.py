# -*- coding: utf-8 -*-
# ============================================================
# 03_q2_dp.py — 问题2：针对 66 岁以上、密集城区保单持有人运行差分隐私
# 拉普拉斯机制: M(D) = f(D) + Lap(Δf/ε)
#   Δf = (upper - lower) / n，Age 截断到 [18, 100]（与讲义一致）
# 对 ε ∈ {0.1, 0.5, 1, 5, 10} 各做 200 次重复发布，画偏差箱型图
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import setup

setup()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
rng = np.random.default_rng(2026)

EPS = [0.1, 0.5, 1.0, 5.0, 10.0]
REPS = 200
LOWER, UPPER = 18, 100


def laplace_noise(scale, size, rng):
    # Lap(0, scale) = -scale * sign(U) * ln(1 - 2|U|), U ~ Uniform(-0.5, 0.5)
    u = rng.uniform(-0.5, 0.5, size)
    return -scale * np.sign(u) * np.log(1 - 2 * np.abs(u))


def dp_mean(x, epsilon, rng, lower=LOWER, upper=UPPER):
    x = np.clip(x, lower, upper)
    n = len(x)
    sens = (upper - lower) / n
    return x.mean() + laplace_noise(sens / epsilon, 1, rng)[0]


df_raw = pd.read_csv(os.path.join(OUT, "df_raw.csv"))

# ---- 三个群体 ----
age = df_raw["Age"].to_numpy()
groups = {
    "整体业务组合 (n=100,000)": age,
    "小型子群体: 男性66+豪华车 (讲义基准)": None,          # 占位，下面填充
    "子群体: 66+ 且密集城区 (本题)": None,
}
mask_sub = (df_raw["AgeGroup"] == "66+") & (df_raw["DensityGroup"] == "Dense urban")
groups["子群体: 66+ 且密集城区 (本题)"] = df_raw.loc[mask_sub, "Age"].to_numpy()
mask_487 = ((df_raw["Gender"] == "Male") & (df_raw["AgeGroup"] == "66+")
            & (df_raw["ValueGroup"] == "Luxury"))
groups["小型子群体: 男性66+豪华车 (讲义基准)"] = df_raw.loc[mask_487, "Age"].to_numpy()

n_sub = int(mask_sub.sum())
n_487 = int(mask_487.sum())
print(f"===== 问题2 =====")
print(f"子群体规模 n (66+ & Dense urban) = {n_sub}")
print(f"男性66+豪华车 n = {n_487} (讲义 487，切点近似)")
print(f"真实平均年龄: 整体 {groups['整体业务组合 (n=100,000)'].mean():.2f}, "
      f"66+密集城区 {groups['子群体: 66+ 且密集城区 (本题)'].mean():.2f}, "
      f"男性66+豪华车 {groups['小型子群体: 男性66+豪华车 (讲义基准)'].mean():.2f}")

# ---- 敏感度 ----
print("\n敏感度 Δf = (100-18)/n：")
for name, x in groups.items():
    print(f"  {name}: Δf = {82/len(x):.5f}")

# ---- 200 次重复发布 ----
rows = []
dev = {}
for name, x in groups.items():
    x = np.clip(x, LOWER, UPPER)
    true_mean = x.mean()
    for eps in EPS:
        est = np.array([dp_mean(x, eps, rng) for _ in range(REPS)])
        d = est - true_mean
        dev[(name, eps)] = d
        rows.append({
            "群体": name, "epsilon": eps, "真实均值": round(true_mean, 2),
            "噪声SD": round(d.std(ddof=1), 3),
            "平均|偏差|": round(np.abs(d).mean(), 3),
            "最大|偏差|": round(np.abs(d).max(), 3),
        })
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "q2_dp_scan.csv"), index=False)
print("\n", res.to_string(index=False))

# ---- 箱型图 ----
fig, ax = plt.subplots(figsize=(9.2, 4.6))
colors = {"整体业务组合 (n=100,000)": "#4C72B0",
          "小型子群体: 男性66+豪华车 (讲义基准)": "#DD8452",
          "子群体: 66+ 且密集城区 (本题)": "#55A868"}
labels_all, data_all, pos_all, cols = [], [], [], []
pos = 0
tick_pos, tick_lab = [], []
for eps in EPS:
    for name in groups:
        pos += 1
        data_all.append(dev[(name, eps)])
        labels_all.append(name)
        cols.append(colors[name])
        pos_all.append(pos)
    tick_pos.append(pos - 1)
    tick_lab.append(f"ε={eps}")
    pos += 1  # 组间空隙
bp = ax.boxplot(data_all, positions=pos_all, widths=0.7, showfliers=False,
                patch_artist=True, medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
ax.set_ylabel("DP 平均年龄估计偏差 (岁)")
ax.set_title(f"不同 ε 下 DP 平均年龄的发布偏差（{REPS} 次重复）："
             f"整体 vs 66+密集城区 (n={n_sub}) vs 男性66+豪华车 (n={n_487})")
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85) for c in colors.values()]
ax.legend(handles, colors.keys(), fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_q2_dp_box.png"), dpi=150)
print("\n图已保存: fig_q2_dp_box.png")

# ---- 理论噪声尺度核对：SD(Lap(b)) = sqrt(2)*b, b = Δf/ε ----
print("\n理论噪声 SD = sqrt(2)*(82/n)/ε 与实测 SD 对照:")
for name, x in groups.items():
    for eps in [0.1, 1.0, 10.0]:
        theo = np.sqrt(2) * (82 / len(x)) / eps
        obs = res[(res["群体"] == name) & (res["epsilon"] == eps)]["噪声SD"].iloc[0]
        print(f"  {name} ε={eps}: 理论 {theo:.3f} vs 实测 {obs:.3f}")
