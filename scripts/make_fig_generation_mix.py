#!/usr/bin/env python3
"""S11: 次世代個体群の構成比（エリート:交叉:注入 = 2:3:1）を示す帯グラフ。

配色は figstyle の3スロット。色だけに頼らないよう各群に直接ラベルを付す。

出力: local/fig_generation_mix.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.set_xlim(0, 11.1)
ax.set_ylim(0, 5.05)
ax.axis("off")

# (ラベル, スロット数, 色, 補足)
GROUPS = [
    ("エリート保存", 2, S.BLUE, "選択個体を無改変で継承"),
    ("交叉 + 微小 Slerp 変異", 3, S.ORANGE, "探索の主力"),
    ("注入", 1, S.AQUA, "初期個体群と同一の手続き"),
]

SLOT_W = 1.66
GAP = 0.10
x = 0.55
y = 2.85
h = 1.00
centers = []
for name, n, col, _ in GROUPS:
    x0 = x
    for _ in range(n):
        ax.add_patch(FancyBboxPatch(
            (x, y), SLOT_W - GAP, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=col, facecolor=S.tint(col, 0.80)))
        x += SLOT_W
    centers.append((x0, x - GAP))
bar_right = x - GAP

# 群のラベルと括弧
for (name, n, col, note), (x0, x1) in zip(GROUPS, centers):
    mid = (x0 + x1) / 2
    ax.add_patch(FancyArrowPatch((x0, y - 0.20), (x1, y - 0.20),
                                 arrowstyle="|-|", mutation_scale=4,
                                 linewidth=1.2, color=col))
    ax.text(mid, y - 0.44, f"{name}（{n}）", ha="center", va="top",
            fontsize=12, fontweight="bold", color=col)
    ax.text(mid, y - 0.92, note, ha="center", va="top", fontsize=10.5,
            color=S.INK_2)

# 見出し
ax.text(0.55, 4.62, "1世代 = 6個体", ha="left", va="center", fontsize=13.5,
        fontweight="bold")
ax.text(0.55, 4.15, "$N = n_e + n_x + n_r = 2 : 3 : 1$", ha="left",
        va="center", fontsize=12.5, color=S.INK_2)

# 提示前のシャッフル（帯の下に1行で置き、右端の注記との衝突を避ける）
ax.add_patch(FancyBboxPatch(
    (0.55, 0.30), bar_right - 0.55, 0.62,
    boxstyle="round,pad=0.02,rounding_size=0.06",
    linewidth=1.1, edgecolor=S.AXIS, facecolor=S.PANEL))
ax.text((0.55 + bar_right) / 2, 0.61,
        "提示前にシャッフルし，操作種別が並び順から読み取れないようにする",
        ha="center", va="center", fontsize=11, color=S.INK)

fig.savefig("local/fig_generation_mix.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_generation_mix.png")
