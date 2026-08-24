# ============================================================
# 02_fit_freq_model.py — 复现讲座 XGBoost 泊松频率模型
# 目标：ClaimNb ~ X (offset: log Exposure)，早停调优，评估 RMSE/Poisson Deviance
# 讲座参考：Best iteration 1325, test score 0.375013
#           RMSE 0.392895, Poisson Deviance 0.487732,
#           Observed Mean 0.149400, Predicted Mean 0.144645
# ============================================================
import os
import time
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")

X_train = pd.read_parquet(os.path.join(OUT, "X_train.parquet"))
X_test = pd.read_parquet(os.path.join(OUT, "X_test.parquet"))
y_train = pd.read_parquet(os.path.join(OUT, "y_train.parquet"))
y_test = pd.read_parquet(os.path.join(OUT, "y_test.parquet"))
ytr, etr = y_train["ClaimNb"].to_numpy(), y_train["Exposure"].to_numpy()
yte, ete = y_test["ClaimNb"].to_numpy(), y_test["Exposure"].to_numpy()

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print(f"训练集平均理赔数: {ytr.mean():.4f}  测试集: {yte.mean():.4f}")

# 训练集内部再切 15% 做早停验证（讲座：早停调优）
rng = np.random.default_rng(123)
n_val = int(0.15 * len(X_train))
val_idx = rng.choice(len(X_train), n_val, replace=False)
tr_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)

dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=ytr[tr_idx],
                  base_margin=np.log(etr[tr_idx]))
dva = xgb.DMatrix(X_train.iloc[val_idx], label=ytr[val_idx],
                  base_margin=np.log(etr[val_idx]))
dte = xgb.DMatrix(X_test, label=yte, base_margin=np.log(ete))

params = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": 8,
    "seed": 42,
}

t0 = time.time()
bst = xgb.train(
    params, dtr, num_boost_round=3000,
    evals=[(dva, "val")],
    early_stopping_rounds=100,
    verbose_eval=200,
)
print(f"训练完成: {time.time()-t0:.1f}s, best_iteration={bst.best_iteration}, best_score={bst.best_score:.6f}")

# ---- 评估 ----
pred_tr = bst.predict(dtr)                    # E[ClaimNb]
pred_te = bst.predict(dte)
# 年化频率 = E[ClaimNb] / Exposure
freq_te = pred_te / ete
freq_obs = yte / ete


def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-12, None)
    return 2.0 * np.mean(y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu))


def rmse(y, mu):
    return np.sqrt(np.mean((y - mu) ** 2))


print("\n===== 频率模型评估（测试集）=====")
print(f"RMSE:              {rmse(yte, pred_te):.6f}")
print(f"Poisson Deviance:  {poisson_deviance(yte, pred_te):.6f}")
print(f"Observed Mean:     {yte.mean():.6f}")
print(f"Predicted Mean:    {pred_te.mean():.6f}")

# 保存模型与预测
with open(os.path.join(OUT, "freq_model.pkl"), "wb") as f:
    pickle.dump({"bst": bst, "features": list(X_train.columns),
                 "params": params,
                 "best_iteration": bst.best_iteration,
                 "best_score": bst.best_score}, f)
np.savez(os.path.join(OUT, "freq_pred.npz"),
         pred_te=pred_te, freq_te=freq_te, freq_obs=freq_obs,
         yte=yte, ete=ete,
         pred_tr=pred_tr, ytr=ytr[tr_idx], etr=etr[tr_idx])
print("模型与预测已保存")
