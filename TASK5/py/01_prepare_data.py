# ============================================================
# 01_prepare_data.py — 第5章数据准备：加载、独热编码、70/30划分
# 目标：复现讲座设计矩阵 (70000, 505)
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "pg15training_ch5.csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)

df = pd.read_csv(DATA)
print("原始 shape:", df.shape)
print("列:", list(df.columns))

# ---- 预测变量（与讲座一致：13 个定价因子 + CalYear 年份？讲座的示例保单含
#      Age/Bonus/Density/Gender/Occupation/Type/Group1/Group2/SubGroup2/
#      Poldur/Value/Category/Adind，共13个）
# 讲座设计矩阵 (70000, 505)：独热编码后 505 列
predictors = ["Age", "Bonus", "Density", "Group1", "Group2", "SubGroup2",
              "Occupation", "Gender", "Poldur", "Type", "Value", "Category", "Adind"]

cat_vars = ["Group1", "Group2", "SubGroup2", "Occupation", "Gender", "Type", "Category", "Adind"]
num_vars = ["Age", "Bonus", "Density", "Poldur", "Value"]

# 分类变量转类别型
for c in cat_vars:
    df[c] = df[c].astype(str)

# 训练/测试 70/30（按行随机，讲座用固定种子划分）
idx = np.random.permutation(len(df))
n_train = int(0.7 * len(df))
train_idx = idx[:n_train]
test_idx = idx[n_train:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# 先划分再独热：训练集独热后做基准，测试集按训练集列对齐（讲座的 align 流程）
for c in cat_vars:
    df_train[c] = df_train[c].astype(str)
    df_test[c] = df_test[c].astype(str)

X_train = pd.get_dummies(df_train[predictors], columns=cat_vars, drop_first=False, dtype=float)
X_test = pd.get_dummies(df_test[predictors], columns=cat_vars, drop_first=False, dtype=float)

# 对齐：测试集补齐缺失列、丢弃多余列
missing = [c for c in X_train.columns if c not in X_test.columns]
extra = [c for c in X_test.columns if c not in X_train.columns]
for c in missing:
    X_test[c] = 0.0
X_test = X_test[X_train.columns]
X_test = X_test[X_train.columns].astype(float)
print(f"训练集独热列数: {X_train.shape[1]}  测试集对齐后列数: {X_test.shape[1]}")
print(f"测试集独有的水平已删除: {len(extra)} 个; 训练集独有水平已补0: {len(missing)} 个")

y_train = df_train["ClaimNb"]
y_test = df_test["ClaimNb"]
exp_train = df_train["Exposure"]
exp_test = df_test["Exposure"]

print("X_train:", X_train.shape, " X_test:", X_test.shape)

# 保存
X_train.to_parquet(os.path.join(OUT, "X_train.parquet"))
X_test.to_parquet(os.path.join(OUT, "X_test.parquet"))
pd.DataFrame({"ClaimNb": y_train, "Exposure": exp_train}).to_parquet(os.path.join(OUT, "y_train.parquet"))
pd.DataFrame({"ClaimNb": y_test, "Exposure": exp_test}).to_parquet(os.path.join(OUT, "y_test.parquet"))

# 原始类别信息（供 group_feature_name 归组）
pd.DataFrame({
    "feature": X_train.columns,
    "group": [g.split("_")[0] for g in X_train.columns],
}).to_csv(os.path.join(OUT, "feature_groups.csv"), index=False)

# 训练/测试保单索引（供后续选择示例保单）
pd.DataFrame({"row": df_train.index.to_numpy()}).to_csv(os.path.join(OUT, "train_rows.csv"), index=False)
pd.DataFrame({"row": df_test.index.to_numpy()}).to_csv(os.path.join(OUT, "test_rows.csv"), index=False)

print("数据准备完成")
