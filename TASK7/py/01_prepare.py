# -*- coding: utf-8 -*-
# ============================================================
# 01_prepare.py — 第7章数据准备：加载 pg15training、构造 df_raw
# 复用讲义的数据加载与准标识符分组方式：
#   Age -> AgeGroup(6档) / Density -> DensityGroup(4档) /
#   Value -> ValueGroup(5档) / Bonus -> BonusGroup(4档)
# 输出 output/df_raw.csv（剥离直接标识符后的 8 列）
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "pg15training_raw.csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
print("原始 shape:", df.shape)

# ---- 讲义的准标识符分组（与讲义 6 条示例记录逐条核对） ----
df["AgeGroup"] = pd.cut(
    df["Age"], [17, 25, 35, 45, 55, 65, 200],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
df["DensityGroup"] = pd.cut(
    df["Density"], [0, 50, 100, 200, np.inf],
    labels=["Rural", "Suburban", "Urban", "Dense urban"])
df["ValueGroup"] = pd.cut(
    df["Value"], [0, 5000, 12000, 28000, 45000, np.inf],
    labels=["Low", "Medium", "High", "Luxury", "Very high"])
df["BonusGroup"] = pd.cut(
    df["Bonus"], [-np.inf, -25, 25, 75, np.inf],
    labels=["Low", "Medium", "High", "Very high"])

df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
df["ClaimAmount"] = df["ClaimTotal"]

df_raw = df[["Gender", "Age", "AgeGroup", "DensityGroup", "ValueGroup",
             "BonusGroup", "HasClaim", "ClaimAmount"]].copy()
df_raw.to_csv(os.path.join(OUT, "df_raw.csv"), index=False)
print("df_raw:", df_raw.shape, "->", os.path.join(OUT, "df_raw.csv"))

# ---- 与讲义锚点核对 ----
print("\n===== 锚点核对 =====")
print("Female n =", (df.Gender == "Female").sum(), "(讲义 36568)")
print("Male n   =", (df.Gender == "Male").sum(), "(讲义 63432)")
print("mean Age =", round(df.Age.mean(), 3), "(讲义 41.13)")
print("claim rate =", round(df.HasClaim.mean(), 4), "(讲义 0.1226)")
m66 = df[(df.Gender == "Male") & (df.Age >= 66)]
print("Male & 66+ =", len(m66), "(讲义 5031)")
print("Male & 66+ & Luxury =", len(m66[m66.ValueGroup == "Luxury"]),
      "(讲义 487，切点为自行标定的近似)")
print("前 6 条记录（对照讲义示例表）:")
print(df_raw.head(6).to_string(index=False))
