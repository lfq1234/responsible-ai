# -*- coding: utf-8 -*-
# ============================================================
# plotstyle.py — 统一 matplotlib 中文字体与样式（同 TASK5）
# ============================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup():
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/simsun.ttc"]:
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
