# ============================================================
# insurance_models.py
# 车险定价案例的完整复刻（讲义 R 代码的 Python 版）：
#   - 数据加载与预处理
#   - 辅助函数（make_age_group / bin_to_factor / jitter / adjust_to_base_portfolio）
#   - GLM：Poisson 频率 + Gamma 严重程度（statsmodels）
#   - XGBoost：count:poisson + reg:gamma（xgboost，base_margin=风险敞口/频率）
#   - MC（模型5）：评分时对性别取平均
#   - 指标：平均保费表、差异性影响比率 DIR、RMSE
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

from fairness_utils import disparate_impact_remover

DATA_PATH = r"C:/Users/LENOVO/Desktop/responsible-ai/TASK3/pg15training_processed.csv"
NON_LEGITIMATE = ["Insurancescore"]
GLM_PREDICTORS = ["Age.ct", "Bonus", "Value", "Density", "GroupOne", "Insurancescore"]
XGB_PREDICTORS = GLM_PREDICTORS
ALL_PREDICTORS_CONT = ["Age.ct", "Bonus.ct", "Value", "Density", "GroupOne.ct", "Insurancescore"]

SEED = 14


def load_claims(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Gender"] = pd.Categorical(df["Gender"], categories=["Male", "Female"])  # relevel ref="Male"
    df["Female"] = (df["Gender"] == "Female").astype(int)
    df["Age"] = pd.Categorical(df["Age"].astype(int).astype(str),
                               categories=[str(i) for i in range(1, 12)])
    df["Bonus"] = pd.Categorical(df["Bonus"].astype(int).astype(str),
                                 categories=[str(i) for i in range(1, 22)])
    df["GroupOne"] = pd.Categorical(df["GroupOne"].astype(int).astype(str),
                                    categories=[str(i) for i in range(1, 21)])
    return df


def claims_with_claims(df):
    """理赔严重程度模型仅基于 Indtppd > 0 的保单。"""
    return df[df["Indtppd"] > 0].copy()


# ---------------- 辅助函数（讲义原样移植） ----------------

def make_age_group(age):
    """连续年龄 -> 年龄段（返回 1..11 的类别编码，对应讲义 cut + as.numeric）。"""
    breaks = [-np.inf, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, np.inf]
    labels = np.arange(1, 12)
    return pd.cut(age, bins=breaks, labels=labels, right=True)


def bin_to_factor(x, min_level, max_level):
    """去偏后的连续值 -> 裁剪 -> 映射回原始整数水平（返回 1-based 编码）。"""
    x = np.asarray(x, dtype=float)
    levels = np.arange(min_level, max_level + 1)
    x = np.clip(x, min_level, max_level)
    cuts = np.concatenate([[min_level - 0.5], levels[1:] - 0.5, [max_level + 0.5]])
    idx = np.searchsorted(cuts, x, side="right") - 1
    idx = np.clip(idx, 0, len(levels) - 1)
    return idx + 1


def jitter_keep_minmax(x, factor=1.0, rng=None):
    """复刻讲义 jitter_keep_minmax()：对非 min/max 观测加均匀噪声（R jitter）。
    噪声幅度 amount = factor/5 * (max-min)。"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    lo, hi = finite.min(), finite.max()
    z = hi - lo
    if z == 0:
        z = abs(lo) if lo != 0 else 1.0
    amount = factor / 5.0 * z
    noise = rng.uniform(-amount, amount, size=len(x))
    keep = (x == lo) | (x == hi)
    out = x + noise
    out[keep] = x[keep]
    return out


def make_di_removed_data(data, features_to_transform, lambda_=1.0, rng=None):
    """复刻讲义 make_di_removed_data()：jitter + DI remover + 重新分段。"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    df = data.copy()
    df["GroupOne.ct"] = df["GroupOne"].cat.codes + 1
    df["Bonus.ct"] = df["Bonus"].cat.codes + 1

    df["Age.ct"] = jitter_keep_minmax(df["Age.ct"], 1, rng)
    df["Bonus.ct"] = jitter_keep_minmax(df["Bonus.ct"], 1, rng)
    df["Value"] = jitter_keep_minmax(df["Value"], 1, rng)
    df["Density"] = jitter_keep_minmax(df["Density"], 1, rng)
    df["GroupOne.ct"] = jitter_keep_minmax(df["GroupOne.ct"], 5, rng)
    df["Insurancescore"] = jitter_keep_minmax(df["Insurancescore"], 1, rng)

    di_df = disparate_impact_remover(df, df["Gender"], features_to_transform,
                                     lambda_=lambda_, rng=rng)

    di_df["Age"] = pd.Categorical(
        make_age_group(di_df["Age.ct"]).astype(int).astype(str),
        categories=[str(i) for i in range(1, 12)])
    di_df["GroupOne"] = pd.Categorical(
        bin_to_factor(di_df["GroupOne.ct"], 1, 20).astype(int).astype(str),
        categories=[str(i) for i in range(1, 21)])
    di_df["Bonus"] = pd.Categorical(
        bin_to_factor(di_df["Bonus.ct"], 1, 21).astype(int).astype(str),
        categories=[str(i) for i in range(1, 22)])
    return di_df


def adjust_to_base_portfolio(raw_premium, base_premium):
    return raw_premium * np.sum(base_premium) / np.sum(raw_premium)


# ---------------- GLM 模型（statsmodels） ----------------

def _design_matrix_glm(df, with_groupone, with_gender):
    """构造与 R 公式一致的 design matrix（treatment coding，基准水平=levels[1]）。"""
    df = df.reset_index(drop=True)
    parts = []
    # 年龄多项式：Age.ct + log(Age.ct) + Age.ct^2..^4
    age = df["Age.ct"].to_numpy(dtype=float)
    parts.append(pd.DataFrame({
        "Age.ct": age,
        "log(Age.ct)": np.log(age),
        "I(Age.ct^2)": age ** 2,
        "I(Age.ct^3)": age ** 3,
        "I(Age.ct^4)": age ** 4,
    }))
    # Bonus 因子（基准 "1"）
    bonus_d = pd.get_dummies(df["Bonus"], prefix="Bonus", drop_first=False)
    bonus_d.columns = [c.replace("Bonus_", "Bonus") for c in bonus_d.columns]
    bonus_d = bonus_d[[f"Bonus{i}" for i in range(2, 22) if f"Bonus{i}" in bonus_d.columns]]
    parts.append(bonus_d.reset_index(drop=True))
    # Density（连续）
    parts.append(pd.DataFrame({"Density": df["Density"].to_numpy(dtype=float)}))
    # GroupOne 因子（仅频率模型）
    if with_groupone:
        g1_d = pd.get_dummies(df["GroupOne"], prefix="GroupOne", drop_first=False)
        g1_d.columns = [c.replace("GroupOne_", "GroupOne") for c in g1_d.columns]
        g1_d = g1_d[[f"GroupOne{i}" for i in range(2, 21) if f"GroupOne{i}" in g1_d.columns]]
        parts.append(g1_d.reset_index(drop=True))
    # Insurancescore（连续）
    parts.append(pd.DataFrame({"Insurancescore": df["Insurancescore"].to_numpy(dtype=float)}))
    # Gender（基准 Male -> GenderFemale 哑变量）
    if with_gender:
        parts.append(pd.DataFrame({"GenderFemale": (df["Gender"] == "Female").astype(int)}))
    X = pd.concat(parts, axis=1).astype(float)
    return X


def fit_glm_pair(df, df_rd, with_gender):
    """拟合频率(Poisson)+严重程度(Gamma) GLM，返回 (freq_model, sev_model)。"""
    X_freq = sm.add_constant(_design_matrix_glm(df, with_groupone=True, with_gender=with_gender))
    offset = np.log(df["Exppdays"].to_numpy(dtype=float))
    freq_model = sm.GLM(df["Numtppd"].to_numpy(dtype=float), X_freq,
                        family=sm.families.Poisson(link=sm.families.links.Log()),
                        offset=offset).fit()

    X_sev = sm.add_constant(_design_matrix_glm(df_rd, with_groupone=False, with_gender=with_gender))
    sev_model = sm.GLM(df_rd["ClaimSeverity"].to_numpy(dtype=float), X_sev,
                       family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    return freq_model, sev_model


def predict_glm_premiums(df, freq_model, sev_model, with_gender):
    """预测纯保费 = 严重程度预测 x 频率预测 x 365。
    freq 预测需除以 Exppdays（因为 offset=log(Exppdays)，且 statsmodels predict
    不会自动带上拟合时的 offset，必须显式传入）。
    with_gender 必须与模型训练时一致（M0=True；MU/MDP/MCDP=False）。"""
    X_freq = sm.add_constant(_design_matrix_glm(df, with_groupone=True, with_gender=with_gender))
    offset = np.log(df["Exppdays"].to_numpy(dtype=float))
    F = freq_model.predict(X_freq, offset=offset) / df["Exppdays"].to_numpy(dtype=float)
    X_sev = sm.add_constant(_design_matrix_glm(df, with_groupone=False, with_gender=with_gender))
    S = sev_model.predict(X_sev)
    return S * F * 365.0


# ---------------- XGBoost 模型 ----------------

def make_xgb_design(df):
    """model.matrix(~0+.)：分类变量全水平哑变量（不设基准），删 GenderMale。"""
    drop = ["Age", "GroupOne.ct", "Bonus.ct", "Female", "ClaimSeverity"]
    cols = [c for c in df.columns if c not in drop]
    d = df[cols]
    # 顺序与 R 一致：因子按其水平全展开（1-based），连续列保持
    gender_d = pd.get_dummies(d["Gender"], prefix="Gender")
    gender_d = gender_d.rename(columns={"Gender_Female": "GenderFemale", "Gender_Male": "GenderMale"})
    bonus_d = pd.get_dummies(d["Bonus"], prefix="Bonus")
    bonus_d.columns = [c.replace("Bonus_", "Bonus") for c in bonus_d.columns]
    g1_d = pd.get_dummies(d["GroupOne"], prefix="GroupOne")
    g1_d.columns = [c.replace("GroupOne_", "GroupOne") for c in g1_d.columns]
    cont = d[["Age.ct", "Density", "Value", "Insurancescore", "Exppdays", "Numtppd", "Indtppd"]]
    out = pd.concat([gender_d, cont, bonus_d, g1_d], axis=1).astype(float)
    if "GenderMale" in out.columns:
        out = out.drop(columns=["GenderMale"])
    return out


def make_xgb_dmatrix(design, response, base_margin, drop_vars):
    x = design.drop(columns=[c for c in drop_vars if c in design.columns])
    dmat = xgb.DMatrix(x.astype(float), label=response if response is not None else None)
    dmat.set_base_margin(np.log(np.asarray(base_margin, dtype=float)))
    return dmat


def align_xgb_design(new_data, train_data):
    missing = [c for c in train_data.columns if c not in new_data.columns]
    for c in missing:
        new_data[c] = 0.0
    return new_data[train_data.columns]


def fit_xgb_freq(xgb_design, nrounds, drop_gender=False, seed=358):
    drop_common = ["Exppdays", "Numtppd", "Indtppd"]
    drop = drop_common + (["GenderFemale"] if drop_gender else [])
    dmat = make_xgb_dmatrix(xgb_design, xgb_design["Numtppd"],
                            xgb_design["Exppdays"], drop)
    params = dict(objective="count:poisson", eval_metric="poisson-nloglik",
                  max_depth=2, eta=0.05, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", seed=seed)
    return xgb.train(params, dmat, num_boost_round=nrounds), dmat


def fit_xgb_sev(xgb_design, nrounds, drop_gender=False, seed=946):
    drop_common = ["Exppdays", "Numtppd", "Indtppd"]
    drop = drop_common + (["GenderFemale"] if drop_gender else [])
    dmat = make_xgb_dmatrix(xgb_design, xgb_design["Indtppd"],
                            xgb_design["Numtppd"], drop)
    params = dict(objective="reg:gamma", max_depth=1, eta=0.015,
                  min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", seed=seed)
    return xgb.train(params, dmat, num_boost_round=nrounds), dmat


# ---------------- 指标 ----------------

def fairness_accuracy_metrics(model_premiums):
    """按 (Method, Model) 计算 male/female 均值、DIR(=male/female)、RMSE。"""
    rows = []
    for (method, model), g in model_premiums.groupby(["Method", "Model"]):
        male = g.loc[g["Gender"] == "Male", "Premium"]
        female = g.loc[g["Gender"] == "Female", "Premium"]
        actual = g["actual"]
        rows.append({
            "Method": method, "Model": model,
            "male_mean": male.mean(), "female_mean": female.mean(),
            "disparate_impact_ratio": male.mean() / female.mean(),
            "rmse": np.sqrt(np.mean((actual - g["Premium"]) ** 2)),
        })
    return pd.DataFrame(rows)


def build_model_premiums(glm_prems: dict, xgb_prems: dict, realclaim):
    """glm_prems / xgb_prems: {model_name: ndarray}；返回长表 DataFrame。"""
    model_order = ["M0: Full Model", "MU: Unawareness Model",
                   "MDP: Demographic Parity", "MCDP: Conditional Demographic Parity",
                   "MC: Controlling for the Protected Variable"]
    frames = []
    for method, prems in (("GLM", glm_prems), ("XGBoost", xgb_prems)):
        for name in model_order:
            frames.append(pd.DataFrame({
                "Method": method, "Model": name,
                "Gender": prems["Gender"], "Premium": prems[name],
                "actual": realclaim,
            }))
    return pd.concat(frames, ignore_index=True)
