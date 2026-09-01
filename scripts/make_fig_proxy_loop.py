#!/usr/bin/env python3
"""質疑用: 自動実験で「人間の何を機械に置き換えたか」を示す図。

本来の IEC と自動実験の違いは選択のステップだけで、次世代の作り方は実装と同一である。
この一点を視覚的に示す。

本来の IEC を BLUE、機械に置き換えた自動実験を ORANGE で示す。

出力: local/fig_proxy_loop.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 6.4)
ax.axis("off")

BW, BH = 3.50, 0.92          # ボックスの幅・高さ
XS = [1.35, 8.35]            # 左右パネルのボックス左端
CH = 0.72                    # 戻り矢印を通す左マージン


def box(x, y, text, ec=S.INK, fc="white", lw=1.4, fs=11.5, ls="-", bold=False,
        tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), BW, BH, boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(x + BW / 2, y + BH / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.25, color=tc or S.INK,
            fontweight="bold" if bold else "normal")


def down(x, y0, y1, color=S.INK):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.4,
                                 color=color))


def loop_back(x_box, y_bot, y_top, color=S.INK):
    """次世代 → 提示 へ戻る矢印。ボックスに重ならないよう左の余白を通す。"""
    xa = x_box - CH
    ax.plot([x_box, xa], [y_bot, y_bot], lw=1.4, color=color, zorder=2)
    ax.plot([xa, xa], [y_bot, y_top], lw=1.4, color=color, zorder=2)
    ax.add_patch(FancyArrowPatch((xa, y_top), (x_box, y_top),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, zorder=2))


PANELS = [("本来の IEC", S.BLUE), ("自動実験（代理適合度）", S.ORANGE)]

for i, (x0, (title, col)) in enumerate(zip(XS, PANELS)):
    # パネル背景
    ax.add_patch(FancyBboxPatch(
        (x0 - 1.15, 0.35), BW + 1.50, 5.35,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor=col, facecolor=S.tint(col, 0.94)))
    ax.text(x0 + BW / 2 - 0.40, 5.98, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=col)

    box(x0, 4.35, "6個体を生成して提示", ec=col)
    down(x0 + BW / 2, 4.28, 3.72, color=col)

    if i == 0:
        box(x0, 2.72, "利用者が\n好みの2体を選択", ec=col,
            fc=S.tint(col, 0.78), lw=2.4, bold=True, tc=col)
    else:
        box(x0, 2.72, "目標プロンプトとの\nコサイン類似度で上位2体", ec=col,
            fc=S.tint(col, 0.78), lw=2.4, ls="--", bold=True, fs=10.5, tc=col)

    down(x0 + BW / 2, 2.65, 2.09, color=col)
    box(x0, 1.05, "エリート / 交叉+変異 / 注入\nで次世代を生成", ec=col, fs=10.5)
    loop_back(x0, 1.51, 4.81, color=col)

# 「違うのはここだけ」
GX0, GX1 = XS[0] + BW + 0.35, XS[1] - CH - 0.35
ax.annotate("", xy=(GX0, 3.18), xytext=(GX1, 3.18),
            arrowprops=dict(arrowstyle="<->", lw=1.8, color=S.INK))
ax.text((GX0 + GX1) / 2, 3.44, "相違は\nここのみ", ha="center",
        va="bottom", fontsize=12, fontweight="bold", linespacing=1.25)
ax.text((GX0 + GX1) / 2, 2.86, "次世代の\n生成手続きは同一", ha="center",
        va="top", fontsize=10, color=S.INK_2, linespacing=1.25)

fig.savefig("local/fig_proxy_loop.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_proxy_loop.png")
