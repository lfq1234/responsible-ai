# ============================================================
# 05_q5_compas.py — B 部分：COMPAS 分类 + roc_pivot theta 扩展扫描
#   1. 加载 fairness 包 compas 数据（已导出 CSV，5278 行两族裔）
#   2. 拟合 M0（含族裔）/ MU（无意识）逻辑回归，复现讲义指标表
#   3. theta = 0.25 / 0.30 扩展 roc_pivot 扫描（连同 0~0.20 复现讲义表）
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairness_utils import roc_pivot, compas_parity_report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

COMPAS_CSV = r"C:/Users/LENOVO/WorkBuddy/2026-08-20-22-11-46/_task3_work/compas_full.csv"

# ---------------- 加载数据（讲义：仅保留两大族裔） ----------------
compas = pd.read_csv(COMPAS_CSV)
compas = compas[compas["ethnicity"].isin(["Caucasian", "African_American"])].copy()
compas["ethnicity"] = pd.Categorical(compas["ethnicity"],
                                     categories=["Caucasian", "African_American"])
compas["Two_yr_Recidivism"] = pd.Categorical(compas["Two_yr_Recidivism"],
                                             categories=["no", "yes"])
print(f"N = {len(compas)}；白人 {sum(compas['ethnicity']=='Caucasian')}；"
      f"非裔美国人 {sum(compas['ethnicity']=='African_American')}")

# 基础再犯率
base_rate = compas.groupby("ethnicity")["Two_yr_Recidivism"].apply(
    lambda s: (s == "yes").mean())
print("基础再犯率:\n", base_rate)

# ---------------- 预测变量（讲义：前科、两个年龄指标、性别、轻罪标志） ----------------
PREDICTORS = ["Number_of_Priors", "Age_Above_FourtyFive", "Age_Below_TwentyFive",
              "Female", "Misdemeanor"]
# factor 转 0/1（no=0, yes=1; Female: Male=0, Female=1）
compas["Age_Above_FourtyFive"] = (compas["Age_Above_FourtyFive"] == "yes").astype(int)
compas["Age_Below_TwentyFive"] = (compas["Age_Below_TwentyFive"] == "yes").astype(int)
compas["Misdemeanor"] = (compas["Misdemeanor"] == "yes").astype(int)
compas["Female"] = (compas["Female"] == "Female").astype(int)
compas["y"] = (compas["Two_yr_Recidivism"] == "yes").astype(int)


def fit_logit(cols):
    X = sm.add_constant(compas[cols])
    return sm.Logit(compas["y"], X).fit(disp=False)


# M0：包含族裔（参考水平 Caucasian，即基准）
compas["ethnicity_AfAm"] = (compas["ethnicity"] == "African_American").astype(int)
model_m0 = fit_logit(PREDICTORS + ["ethnicity_AfAm"])
# MU：移除族裔
model_mu = fit_logit(PREDICTORS)

p_recid_m0 = model_m0.predict(sm.add_constant(compas[PREDICTORS + ["ethnicity_AfAm"]]))
p_recid_mu = model_mu.predict(sm.add_constant(compas[PREDICTORS]))

pred_m0 = np.where(p_recid_m0 > 0.5, "yes", "no")
pred_mu = np.where(p_recid_mu > 0.5, "yes", "no")

# ---------------- 复现讲义指标表（theta=0 即 M0/MU 未修正） ----------------
print("\n===== M0 / MU 指标（theta = 0）=====")
for name, p, pred in [("M0", p_recid_m0, pred_m0), ("MU", p_recid_mu, pred_mu)]:
    r = compas_parity_report(compas["Two_yr_Recidivism"], pred, compas["ethnicity"])
    print(f"\n[{name}] 准确率={r['accuracy']:.4f}")
    print(f"  被标记高风险: 白人 {r['flagged_white']:.4f} / 非裔 {r['flagged_African_American']:.4f}")
    print(f"  FPR: 白人 {r['fpr_white']:.4f} / 非裔 {r['fpr_African_American']:.4f}")
    print(f"  FNR: 白人 {r['fnr_white']:.4f} / 非裔 {r['fnr_African_American']:.4f}")
    print(f"  精确率: 白人 {r['precision_white']:.4f} / 非裔 {r['precision_African_American']:.4f}")
    print(f"  人口统计学均等性={r['prop_parity']:.3f}  TPR差距={r['equal_odds']:.3f}  预测率均等性={r['pred_rate_parity']:.3f}")

# ---------------- roc_pivot：theta 扫描（0 ~ 0.30） ----------------
# 讲义：高于 0.5 的概率=再犯(不利)，因此先转为"预测不再犯"(有利)，白人=优势群体
print("\n===== roc_pivot theta 扫描（基于 MU）=====")
theta_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
scan_rows = []
for theta in theta_list:
    if theta == 0.0:
        p_norecid_fixed = 1.0 - p_recid_mu.to_numpy()
    else:
        p_norecid_fixed = roc_pivot(1.0 - p_recid_mu.to_numpy(),
                                    compas["ethnicity"].to_numpy(),
                                    privileged="Caucasian", cutoff=0.5, theta=theta)
    pred_fixed = np.where(p_norecid_fixed > 0.5, "no", "yes")  # 修正后的"再犯"标签
    r = compas_parity_report(compas["Two_yr_Recidivism"], pred_fixed, compas["ethnicity"])
    row = {"theta": theta, "accuracy": r["accuracy"],
           "prop_parity": r["prop_parity"], "equal_odds": r["equal_odds"],
           "pred_rate_parity": r["pred_rate_parity"],
           "flagged_white": r["flagged_white"], "flagged_AfAm": r["flagged_African_American"]}
    scan_rows.append(row)
    print(f"theta={theta:>4}: 准确率={row['accuracy']:.4f}  人口统计学均等性={row['prop_parity']:.3f}  "
          f"TPR差距={row['equal_odds']:.3f}  预测率均等性={row['pred_rate_parity']:.3f}  "
          f"(被标高风险 白人 {row['flagged_white']:.3f} / 非裔 {row['flagged_AfAm']:.3f})")

scan_df = pd.DataFrame(scan_rows)
scan_df.to_csv(os.path.join(OUT, "compas_theta_scan.csv"), index=False)

with open(os.path.join(OUT, "compas_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "N": int(len(compas)),
        "n_white": int((compas["ethnicity"] == "Caucasian").sum()),
        "n_afam": int((compas["ethnicity"] == "African_American").sum()),
        "base_rate_white": float(base_rate["Caucasian"]),
        "base_rate_afam": float(base_rate["African_American"]),
        "m0": compas_parity_report(compas["Two_yr_Recidivism"], pred_m0, compas["ethnicity"]),
        "mu": compas_parity_report(compas["Two_yr_Recidivism"], pred_mu, compas["ethnicity"]),
        "theta_scan": scan_rows,
    }, f, ensure_ascii=False, indent=2)
print("\nCOMPAS 完成，结果已保存")
