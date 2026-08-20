# ============================================================
# 01_q1_q2.py — A 部分编程题
#   问题1：MDP 的 lambda 扫描（lambda = 0, 0.25, 0.5, 0.75, 1）
#   问题2：仅对 Density 去偏（lambda = 1）
# 两者均用 GLM（Poisson 频率 + Gamma 严重程度）重新拟合，
# 计算差异性影响比率 DIR（male/female）与 RMSE。
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insurance_models import (
    load_claims, claims_with_claims, make_di_removed_data,
    fit_glm_pair, _design_matrix_glm, SEED,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)


def adjust_to_base(raw, base):
    return np.asarray(raw, dtype=float) * np.sum(base) / np.sum(np.asarray(raw, dtype=float))


def predict_freq(df, model, with_gender):
    X = sm.add_constant(_design_matrix_glm(df, with_groupone=True, with_gender=with_gender))
    offset = np.log(df["Exppdays"].to_numpy(dtype=float))
    return model.predict(X, offset=offset) / df["Exppdays"].to_numpy(dtype=float)


def predict_sev(df, model, with_gender):
    X = sm.add_constant(_design_matrix_glm(df, with_groupone=False, with_gender=with_gender))
    return model.predict(X)


def run_design(data_name, df, df_rd, with_gender):
    """拟合 GLM 并返回 (DIR, RMSE, mean_table) 与组合调整后的预测。"""
    freq, sev = fit_glm_pair(df, df_rd, with_gender=with_gender)
    S = predict_sev(df, sev, with_gender)
    F = predict_freq(df, freq, with_gender)
    prem = S * F * 365
    actual = df["Indtppd"].to_numpy(dtype=float) / df["Exppdays"].to_numpy(dtype=float) * 365
    male = prem[df["Gender"].to_numpy() == "Male"]
    female = prem[df["Gender"].to_numpy() == "Female"]
    dir_ratio = male.mean() / female.mean()
    rmse = np.sqrt(np.mean((actual - prem) ** 2))
    return dict(dir=dir_ratio, rmse=rmse, male_mean=male.mean(), female_mean=female.mean(),
                total=prem.sum())


ClaimsData = load_claims()
ClaimsData_rd = claims_with_claims(ClaimsData)

# ============================================================
# 问题1：lambda 扫描（全预测变量去偏）
# ============================================================
print("===== 问题1：MDP lambda 扫描（全变量去偏）=====")
ALL_VARS = ["Age.ct", "Bonus.ct", "Value", "Density", "GroupOne.ct", "Insurancescore"]
q1_results = []
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    debiased = make_di_removed_data(ClaimsData, ALL_VARS, lambda_=lam, rng=rng)
    debiased_rd = claims_with_claims(debiased)
    r = run_design(f"lambda={lam}", debiased, debiased_rd, with_gender=False)
    r["lambda"] = lam
    q1_results.append(r)
    print(f"lambda={lam:>4}: DIR={r['dir']:.4f}  RMSE={r['rmse']:.4f}  "
          f"male={r['male_mean']:.2f}  female={r['female_mean']:.2f}  total={r['total']:.0f}")

q1_df = pd.DataFrame(q1_results)
q1_df.to_csv(os.path.join(OUT, "q1_lambda_scan.csv"), index=False)

# ============================================================
# 问题2：仅对 Density 去偏（lambda=1），与问题1 lambda=0（MU）比较
# ============================================================
print("\n===== 问题2：仅对 Density 去偏（lambda=1）=====")
deb_density = make_di_removed_data(ClaimsData, ["Density"], lambda_=1.0, rng=rng)
deb_density_rd = claims_with_claims(deb_density)
q2 = run_design("Density-only", deb_density, deb_density_rd, with_gender=False)
print(f"仅去偏 Density: DIR={q2['dir']:.4f}  RMSE={q2['rmse']:.4f}  "
      f"male={q2['male_mean']:.2f}  female={q2['female_mean']:.2f}")

# 参考：M0（含 Gender）与 MU（无 Gender）的 GLM 结果
m0 = run_design("M0", ClaimsData, ClaimsData_rd, with_gender=True)
mu = run_design("MU", ClaimsData, ClaimsData_rd, with_gender=False)
print(f"\n参考基线 GLM:")
print(f"  M0: DIR={m0['dir']:.4f}  RMSE={m0['rmse']:.4f}  male={m0['male_mean']:.2f}  female={m0['female_mean']:.2f}")
print(f"  MU: DIR={mu['dir']:.4f}  RMSE={mu['rmse']:.4f}  male={mu['male_mean']:.2f}  female={mu['female_mean']:.2f}")

summary = {
    "q1_lambda_scan": q1_results,
    "q2_density_only": q2,
    "reference_M0": m0,
    "reference_MU": mu,
}
with open(os.path.join(OUT, "q1_q2_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n问题1/2 完成")
