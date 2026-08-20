# ============================================================
# 00_baseline.py — 复现讲义基线：五种模型设计 x GLM/XGBoost
# 输出：平均保费表、DIR、RMSE；保存模型与预测供问题1/2 使用
# ============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insurance_models import (
    load_claims, claims_with_claims, make_di_removed_data,
    fit_glm_pair, predict_glm_premiums,
    make_xgb_design, fit_xgb_freq, fit_xgb_sev,
    align_xgb_design, make_xgb_dmatrix, _design_matrix_glm,
    build_model_premiums, fairness_accuracy_metrics,
    NON_LEGITIMATE, SEED,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)


def adjust_to_base(raw, base):
    """按比例缩放，使总额与基准组合一致（讲义 adjust_to_base_portfolio）。"""
    raw = np.asarray(raw, dtype=float)
    return raw * np.sum(base) / np.sum(raw)


def predict_freq(df, model, with_gender):
    X = sm.add_constant(_design_matrix_glm(df, with_groupone=True, with_gender=with_gender))
    offset = np.log(df["Exppdays"].to_numpy(dtype=float))
    return model.predict(X, offset=offset) / df["Exppdays"].to_numpy(dtype=float)


def predict_sev(df, model, with_gender):
    X = sm.add_constant(_design_matrix_glm(df, with_groupone=False, with_gender=with_gender))
    return model.predict(X)


def score_freq(model, design, drop_gender):
    drop = ["Exppdays", "Numtppd", "Indtppd"] + (["GenderFemale"] if drop_gender else [])
    dmat = make_xgb_dmatrix(design, None, design["Exppdays"], drop)
    return model.predict(dmat) / design["Exppdays"].to_numpy(dtype=float) * 365


def score_sev(model, design, base_margin, drop_gender):
    drop = ["Exppdays", "Numtppd", "Indtppd"] + (["GenderFemale"] if drop_gender else [])
    dmat = make_xgb_dmatrix(design, None, base_margin, drop)
    return model.predict(dmat)


# ---------------- 数据 ----------------
ClaimsData = load_claims()
ClaimsData_rd = claims_with_claims(ClaimsData)
print(f"样本量: {len(ClaimsData)}；理赔子集: {len(ClaimsData_rd)}")

# ---------------- 去偏数据集（讲义顺序：MDP 全去偏 -> MCDP 仅 Insurancescore） ----------------
ClaimsData_M3 = make_di_removed_data(
    ClaimsData, ["Age.ct", "Bonus.ct", "Value", "Density", "GroupOne.ct", "Insurancescore"],
    lambda_=1.0, rng=rng)
ClaimsData_M4 = make_di_removed_data(ClaimsData, ["Insurancescore"], lambda_=1.0, rng=rng)
ClaimsData_M3_rd = claims_with_claims(ClaimsData_M3)
ClaimsData_M4_rd = claims_with_claims(ClaimsData_M4)

# ---------------- GLM：M0/MU/MDP/MCDP ----------------
print("拟合 GLM（M0/MU/MDP/MCDP）...")
freq1, sev1 = fit_glm_pair(ClaimsData, ClaimsData_rd, with_gender=True)            # M0
freq2, sev2 = fit_glm_pair(ClaimsData, ClaimsData_rd, with_gender=False)           # MU
freq3, sev3 = fit_glm_pair(ClaimsData_M3, ClaimsData_M3_rd, with_gender=False)     # MDP
freq4, sev4 = fit_glm_pair(ClaimsData_M4, ClaimsData_M4_rd, with_gender=False)     # MCDP
print("GLM 拟合完成")

# MC（模型5）：评分时对性别取平均（复用 M0 系数）
gender_coef_sev  = sev1.params["GenderFemale"]
gender_coef_freq = freq1.params["GenderFemale"]

# ---------------- GLM 预测（五种设计） ----------------
sev_pred = {1: predict_sev(ClaimsData, sev1, True), 2: predict_sev(ClaimsData, sev2, False),
            3: predict_sev(ClaimsData_M3, sev3, False), 4: predict_sev(ClaimsData_M4, sev4, False)}
freq_pred = {1: predict_freq(ClaimsData, freq1, True), 2: predict_freq(ClaimsData, freq2, False),
             3: predict_freq(ClaimsData_M3, freq3, False), 4: predict_freq(ClaimsData_M4, freq4, False)}

female = ClaimsData["Female"].to_numpy(dtype=float)
S_if_female = np.where(female == 1, sev_pred[1], sev_pred[1] * np.exp(gender_coef_sev))
S_if_male   = np.where(female == 0, sev_pred[1], sev_pred[1] * np.exp(-gender_coef_sev))
S5 = (S_if_female + S_if_male) / 2
F_if_female = np.where(female == 1, freq_pred[1], freq_pred[1] * np.exp(gender_coef_freq))
F_if_male   = np.where(female == 0, freq_pred[1], freq_pred[1] * np.exp(-gender_coef_freq))
F5 = (F_if_female + F_if_male) / 2

raw_prem = pd.DataFrame({
    "PurePrem1": sev_pred[1] * freq_pred[1] * 365,
    "PurePrem2": sev_pred[2] * freq_pred[2] * 365,
    "PurePrem3_raw": sev_pred[3] * freq_pred[3] * 365,
    "PurePrem4_raw": sev_pred[4] * freq_pred[4] * 365,
    "PurePrem5_raw": S5 * F5 * 365,
})
glmpred_sum = pd.concat([ClaimsData.reset_index(drop=True), raw_prem], axis=1)
glmpred_sum["PurePrem3"] = adjust_to_base(glmpred_sum["PurePrem3_raw"], glmpred_sum["PurePrem2"])
glmpred_sum["PurePrem4"] = adjust_to_base(glmpred_sum["PurePrem4_raw"], glmpred_sum["PurePrem2"])
glmpred_sum["PurePrem5"] = adjust_to_base(glmpred_sum["PurePrem5_raw"], glmpred_sum["PurePrem2"])
glmpred_sum["realclaim"] = glmpred_sum["Indtppd"] / glmpred_sum["Exppdays"] * 365

# ---------------- XGBoost ----------------
print("构建 XGBoost 设计矩阵...")
xgbData    = make_xgb_design(ClaimsData)
xgbData_M3 = make_xgb_design(ClaimsData_M3)
xgbData_M4 = xgbData.copy()
m4_vars = [c for c in NON_LEGITIMATE if c in xgbData_M4.columns]
xgbData_M4[m4_vars] = xgbData_M3[m4_vars]

print("训练 XGBoost 频率模型（约 5-10 分钟）...")
xgbFreq1, _ = fit_xgb_freq(xgbData, 3958, drop_gender=False)
xgbFreq2, _ = fit_xgb_freq(xgbData, 3988, drop_gender=True)
xgbFreq3, _ = fit_xgb_freq(xgbData_M3, 3151, drop_gender=True)
xgbFreq4, _ = fit_xgb_freq(xgbData_M4, 3987, drop_gender=True)

print("训练 XGBoost 严重程度模型...")
xgbData_sev    = xgbData[xgbData["Indtppd"] > 0].reset_index(drop=True)
xgbData_M3_sev = xgbData_M3[xgbData_M3["Indtppd"] > 0].reset_index(drop=True)
xgbData_M4_sev = xgbData_M4[xgbData_M4["Indtppd"] > 0].reset_index(drop=True)
xgbSev1, _ = fit_xgb_sev(xgbData_sev, 2239, drop_gender=False)
xgbSev2, _ = fit_xgb_sev(xgbData_sev, 2113, drop_gender=True)
xgbSev3, _ = fit_xgb_sev(xgbData_M3_sev, 2333, drop_gender=True)
xgbSev4, _ = fit_xgb_sev(xgbData_M4_sev, 1908, drop_gender=True)

# ---------------- XGBoost 评分 ----------------
xgb_pred = align_xgb_design(make_xgb_design(ClaimsData), xgbData)
xgb_pred_debiased = align_xgb_design(make_xgb_design(ClaimsData_M3), xgbData_M3)

xgb_pred_M3 = xgb_pred.copy()
m3_vars = [c for c in (["Age.ct", "Value", "Density", "Insurancescore"] +
                       [c for c in xgb_pred_M3.columns if c.startswith("GroupOne")] +
                       [c for c in xgb_pred_M3.columns if c.startswith("Bonus")])
           if c in xgb_pred_M3.columns]
xgb_pred_M3[m3_vars] = xgb_pred_debiased[m3_vars]
xgb_pred_M3 = align_xgb_design(xgb_pred_M3, xgbData_M3)

xgb_pred_M4 = xgb_pred.copy()
xgb_pred_M4[m4_vars] = xgb_pred_M3[m4_vars]
xgb_pred_M4 = align_xgb_design(xgb_pred_M4, xgbData_M4)

xgb_pred_rev = xgb_pred.copy()
xgb_pred_rev["GenderFemale"] = 1 - xgb_pred_rev["GenderFemale"]

xgbFreq1_pred = score_freq(xgbFreq1, xgb_pred, False)
xgbFreq2_pred = score_freq(xgbFreq2, xgb_pred, True)
xgbFreq3_pred = score_freq(xgbFreq3, xgb_pred_M3, True)
xgbFreq4_pred = score_freq(xgbFreq4, xgb_pred_M4, True)
xgbFreq5_pred = score_freq(xgbFreq1, xgb_pred_rev, False)

xgbPrem1_raw = score_sev(xgbSev1, xgb_pred, xgbFreq1_pred, False)
xgbPrem2_raw = score_sev(xgbSev2, xgb_pred, xgbFreq2_pred, True)
xgbPrem3_raw = score_sev(xgbSev3, xgb_pred_M3, xgbFreq3_pred, True)
xgbPrem4_raw = score_sev(xgbSev4, xgb_pred_M4, xgbFreq4_pred, True)
xgbPrem5_raw = 0.5 * (score_sev(xgbSev1, xgb_pred, xgbFreq1_pred, False) +
                      score_sev(xgbSev1, xgb_pred_rev, xgbFreq5_pred, False))

# 组合层面调整：所有 XGBoost 总额对齐 GLM MU（PurePrem2）
total_glm_mu = glmpred_sum["PurePrem2"].sum()
xgbpred_sum = ClaimsData.reset_index(drop=True).copy()
for i, raw in enumerate([xgbPrem1_raw, xgbPrem2_raw, xgbPrem3_raw, xgbPrem4_raw, xgbPrem5_raw], 1):
    xgbpred_sum[f"xgbPurePrem{i}"] = raw / raw.sum() * total_glm_mu
xgbpred_sum["realclaim"] = xgbpred_sum["Indtppd"] / xgbpred_sum["Exppdays"] * 365

# ---------------- 指标 ----------------
glm_prems = {"Gender": ClaimsData["Gender"].to_numpy(),
             "M0: Full Model": glmpred_sum["PurePrem1"].to_numpy(),
             "MU: Unawareness Model": glmpred_sum["PurePrem2"].to_numpy(),
             "MDP: Demographic Parity": glmpred_sum["PurePrem3"].to_numpy(),
             "MCDP: Conditional Demographic Parity": glmpred_sum["PurePrem4"].to_numpy(),
             "MC: Controlling for the Protected Variable": glmpred_sum["PurePrem5"].to_numpy()}
xgb_prems = {"Gender": ClaimsData["Gender"].to_numpy(),
             "M0: Full Model": xgbpred_sum["xgbPurePrem1"].to_numpy(),
             "MU: Unawareness Model": xgbpred_sum["xgbPurePrem2"].to_numpy(),
             "MDP: Demographic Parity": xgbpred_sum["xgbPurePrem3"].to_numpy(),
             "MCDP: Conditional Demographic Parity": xgbpred_sum["xgbPurePrem4"].to_numpy(),
             "MC: Controlling for the Protected Variable": xgbpred_sum["xgbPurePrem5"].to_numpy()}

realclaim = ClaimsData["Indtppd"].to_numpy() / ClaimsData["Exppdays"].to_numpy() * 365
model_premiums = build_model_premiums(glm_prems, xgb_prems, realclaim)
fa = fairness_accuracy_metrics(model_premiums)

mean_table = model_premiums.groupby(["Method", "Gender", "Model"])["Premium"].mean().unstack()
print("\n===== 平均预测纯保费（按性别）=====")
print(mean_table.round(2).to_string())
mean_table.round(2).to_csv(os.path.join(OUT, "mean_premium_table.csv"))

print("\n===== 公平性—准确性汇总 =====")
print(fa.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
fa.to_csv(os.path.join(OUT, "fairness_accuracy.csv"), index=False)

with open(os.path.join(OUT, "baseline_results.pkl"), "wb") as f:
    pickle.dump({
        "ClaimsData": ClaimsData, "ClaimsData_M3": ClaimsData_M3,
        "ClaimsData_M4": ClaimsData_M4,
        "glmpred_sum": glmpred_sum, "xgbpred_sum": xgbpred_sum,
        "fairness_accuracy": fa, "mean_table": mean_table,
        "gender_coef_sev": gender_coef_sev, "gender_coef_freq": gender_coef_freq,
        "freq1": freq1, "sev1": sev1, "freq2": freq2, "sev2": sev2,
        "freq3": freq3, "sev3": sev3, "freq4": freq4, "sev4": sev4,
        "glm_prems": glm_prems, "xgb_prems": xgb_prems, "realclaim": realclaim,
    }, f)
print("\n基线完成，结果已保存到", OUT)
