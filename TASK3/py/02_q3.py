# ============================================================
# 02_q3.py — 问题3：为何仅靠无意识是不够的
# 利用讲义中的平均保费表计算 M0 与 MU 的差异性影响比率（DIR）
# ============================================================

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# 讲义：按性别划分的平均预测纯保费表
# 列: M0, MU, MDP, MCDP, MC
mean_premiums = {
    "GLM": {
        "male":   {"M0": 130.47, "MU": 114.03, "MDP": 117.99, "MCDP": 115.25, "MC": 113.95},
        "female": {"M0": 95.66,  "MU": 124.05, "MDP": 117.17, "MCDP": 121.92, "MC": 124.18},
    },
    "XGBoost": {
        "male":   {"M0": 130.98, "MU": 114.41, "MDP": 118.36, "MCDP": 116.59, "MC": 114.23},
        "female": {"M0": 94.63,  "MU": 123.39, "MDP": 116.54, "MCDP": 119.60, "MC": 123.69},
    },
}

# DIR = male_mean / female_mean（与讲义代码一致）
print("===== 问题3a：讲义平均保费表的 DIR（male/female）=====")
results = {}
for method in ["GLM", "XGBoost"]:
    for model in ["M0", "MU", "MDP", "MCDP", "MC"]:
        male = mean_premiums[method]["male"][model]
        female = mean_premiums[method]["female"][model]
        dir_ = male / female
        results[f"{method}_{model}"] = {"male": male, "female": female, "DIR": dir_}
        tag = ""
        if model == "M0":
            tag = "  <- 女性明显更低（男性支付更多）"
        elif model == "MU":
            tag = "  <- 反转：女性反而更高（男性支付更少）"
        print(f"{method:>8} {model:>4}: male={male:7.2f}  female={female:7.2f}  DIR={dir_:.3f}{tag}")

with open(os.path.join(OUT, "q3_dir.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n问题3a 计算完成")
