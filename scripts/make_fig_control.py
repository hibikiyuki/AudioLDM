#!/usr/bin/env python3
"""S8: 初期ノイズを固定する理由（統制）の対比図を生成する。

候補が意味と音色の両方で同時に散らばると、利用者の選択がどちらへの判断か分離できない。
x_T を固定すると候補間の差が意味の側だけになり、選択が意味軸への信号として機能する。

意味軸を BLUE、テクスチャ軸を ORANGE とし、この対応は発表全体で固定する。

出力: local/fig_control.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import figstyle as S

S.use(12)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3))
fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.02, wspace=0.18)


def frame(ax, title):
    """共通の軸（意味 × 音響テクスチャ）を描く。"""
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.66, 1.32)
    ax.axis("off")
    ax.add_patch(FancyArrowPatch((0, 0), (1.12, 0), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.3, color=S.ORANGE))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.14), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.3, color=S.BLUE))
    ax.text(1.14, -0.05, "音響テクスチャ ($x_T$)", ha="right", va="top",
            fontsize=10, color=S.ORANGE, fontweight="bold")
    ax.text(-0.02, 1.19, "意味 ($c$)", ha="left", va="bottom", fontsize=10,
            color=S.BLUE, fontweight="bold")
    ax.text(0.49, 1.42, title, ha="center", va="center", fontsize=13,
            fontweight="bold", transform=ax.transData)


def points(ax, xs, ys, sel):
    """候補を描く。sel の個体だけ塗りつぶす（＝利用者が選んだもの）。"""
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == sel:
            ax.plot(x, y, "o", ms=12, mfc=S.BLUE, mec=S.BLUE, mew=1.4,
                    zorder=3)
        else:
            ax.plot(x, y, "o", ms=10, mfc="white", mec=S.INK_2, mew=1.4,
                    zorder=3)


def caption(ax, text, emph=False):
    ax.add_patch(FancyBboxPatch(
        (-0.16, -0.60), 1.34, 0.28,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4 if emph else 1.1,
        edgecolor=S.BLUE if emph else S.AXIS,
        facecolor=S.tint(S.BLUE, 0.88) if emph else "white",
        linestyle="-" if emph else "--",
        transform=ax.transData, clip_on=False, zorder=1))
    ax.text(0.51, -0.46, text, ha="center", va="center", fontsize=11,
            color=S.BLUE if emph else S.INK,
            fontweight="bold" if emph else "normal", linespacing=1.35)


# ---- 左: x_T を固定しない ----
ax = axes[0]
frame(ax, "$x_T$ を固定しない")
# 候補は手で配置する（重なりを避け、両軸に散っていることを明示するため）
xs = np.array([0.17, 0.33, 0.49, 0.66, 0.83, 0.97])
ys = np.array([0.34, 0.70, 0.15, 0.93, 0.50, 0.24])
SEL_L = 3
points(ax, xs, ys, sel=SEL_L)
ax.plot([xs[SEL_L], xs[SEL_L]], [0, ys[SEL_L]], ls=":", lw=1.3, color=S.ORANGE,
        zorder=1)
ax.plot([0, xs[SEL_L]], [ys[SEL_L], ys[SEL_L]], ls=":", lw=1.3, color=S.BLUE,
        zorder=1)
ax.text(0.30, 1.05, "候補が両軸に分散", ha="center", va="center", fontsize=10.5,
        color=S.INK_2)
caption(ax, "選択の由来が意味か音色か\n分離できない")

# ---- 右: x_T を固定する ----
ax = axes[1]
frame(ax, "$x_T$ を固定する")
x_fixed = 0.52
ax.add_patch(Rectangle((x_fixed - 0.055, 0), 0.11, 1.02,
                       facecolor=S.tint(S.ORANGE, 0.86), edgecolor="none",
                       zorder=0))
ax.plot([x_fixed, x_fixed], [0, 1.02], ls="--", lw=1.4, color=S.ORANGE,
        zorder=1)
ys2 = np.array([0.15, 0.31, 0.47, 0.62, 0.78, 0.94])
SEL_R = 4
points(ax, np.full(6, x_fixed), ys2, sel=SEL_R)
ax.plot([0, x_fixed], [ys2[SEL_R], ys2[SEL_R]], ls=":", lw=1.3, color=S.BLUE,
        zorder=1)
ax.annotate("音色は全候補で共通", xy=(x_fixed + 0.05, 1.00), xytext=(1.00, 1.12),
            ha="center", va="bottom", fontsize=10, color=S.ORANGE,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=S.ORANGE,
                            connectionstyle="arc3,rad=0.25"))
caption(ax, "差異は意味の側のみ\n→ 選択が意味軸への信号となる", emph=True)

fig.savefig("local/fig_control.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_control.png")
