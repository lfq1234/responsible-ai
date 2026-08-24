# ============================================================
# 02b_fit_freq_model_nolink.py — 无 offset 泊松频率模型（供 TreeSHAP 使用）
# label = ClaimNb / Exposure（年化索赔率），objective=count:poisson
# 预测值 = 年化索赔频率；log(频率) = 树输出之和，TreeSHAP 加法性精确成立
# 评估仍以 counts 尺度（RMSE / Poisson Deviance）与讲座对比
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

# 年化频率 label（泊松速率）
rate_tr = ytr / etr
rate_te = yte / ete
print("rate train mean:", rate_tr.mean().round(5), " test mean:", rate_te.mean().round(5))

rng = np.random.default_rng(123)
n_val = int(0.15 * len(X_train))
val_idx = rng.choice(len(X_train), n_val, replace=False)
tr_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)

dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=rate_tr[tr_idx])
dva = xgb.DMatrix(X_train.iloc[val_idx], label=rate_tr[val_idx])
dte = xgb.DMatrix(X_test, label=rate_te)

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
bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")],
                early_stopping_rounds=100, verbose_eval=200)
print(f"训练完成: {time.time()-t0:.1f}s, best_iteration={bst.best_iteration}, best_score={bst.best_score:.6f}")

# 预测（年化频率）
rate_pred = bst.predict(dte)
pred_counts = rate_pred * ete


def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-12, None)
    return 2.0 * np.mean(y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu))


def rmse(y, mu):
    return np.sqrt(np.mean((y - mu) ** 2))


print("\n===== 无 offset 频率模型评估（测试集, counts 尺度）=====")
print(f"RMSE:              {rmse(yte, pred_counts):.6f}")
print(f"Poisson Deviance:  {poisson_deviance(yte, pred_counts):.6f}")
print(f"Observed Mean:     {yte.mean():.6f}")
print(f"Predicted Mean:    {pred_counts.mean():.6f}")

with open(os.path.join(OUT, "freq_model_nolink.pkl"), "wb") as f:
    pickle.dump({"bst": bst, "features": list(X_train.columns),
                 "params": params,
                 "best_iteration": bst.best_iteration,
                 "best_score": bst.best_score}, f)
np.savez(os.path.join(OUT, "freq_pred_nolink.npz"),
         rate_pred=rate_pred, pred_counts=pred_counts,
         yte=yte, ete=ete, rate_te=rate_te)
print("模型与预测已保存")
