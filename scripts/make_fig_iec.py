#!/usr/bin/env python3
"""対話型進化計算 (IEC) とは何かを、通常の進化計算との差分で示す図。

IEC の要点は「適合度を数式で書けないときに、人間を評価器として組み込む」ことに尽きる。
ループの形は通常の進化計算と同じで、評価のステップだけが違う。

出力: local/fig_iec.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 6.8)
ax.axis("off")

BW, BH = 3.60, 0.92
XS = [1.40, 8.30]
CH = 0.74
PANELS = [("通常の進化計算", S.MUTED), ("対話型進化計算 (IEC)", S.BLUE)]


def box(x, y, text, ec=S.INK, fc="white", lw=1.4, fs=11.5, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), BW, BH, boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + BW / 2, y + BH / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.25, color=tc or S.INK,
            fontweight="bold" if bold else "normal")


def down(x, y0, y1, color):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.4, color=color))


def loop_back(x_box, y_bot, y_top, color):
    xa = x_box - CH
    ax.plot([x_box, xa], [y_bot, y_bot], lw=1.4, color=color, zorder=2)
    ax.plot([xa, xa], [y_bot, y_top], lw=1.4, color=color, zorder=2)
    ax.add_patch(FancyArrowPatch((xa, y_top), (x_box, y_top), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.4, color=color,
                                 zorder=2))


for i, (x0, (title, col)) in enumerate(zip(XS, PANELS)):
    ax.add_patch(FancyBboxPatch(
        (x0 - 1.18, 0.90), BW + 1.55, 5.05,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor=col, facecolor=S.tint(col, 0.94)))
    ax.text(x0 + BW / 2 - 0.40, 6.28, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=col)

    box(x0, 4.62, "候補を生成", ec=col)
    down(x0 + BW / 2, 4.55, 4.02, col)

    if i == 0:
        box(x0, 2.98, "適合度 $f(x)$ で評価\n（式で定義できる）", ec=col,
            fc=S.tint(col, 0.78), lw=2.4, bold=True, tc=S.INK)
    else:
        box(x0, 2.98, "人が聴いて選択\n（式で定義できない）", ec=col,
            fc=S.tint(col, 0.78), lw=2.4, bold=True, tc=col)

    down(x0 + BW / 2, 2.91, 2.38, col)
    box(x0, 1.34, "良い個体を親として\n次世代を生成", ec=col, fs=11)
    loop_back(x0, 1.80, 5.08, col)

# 「違うのは評価だけ」
GX0, GX1 = XS[0] + BW + 0.32, XS[1] - CH - 0.32
ax.annotate("", xy=(GX0, 3.44), xytext=(GX1, 3.44),
            arrowprops=dict(arrowstyle="<->", lw=1.8, color=S.INK))
ax.text((GX0 + GX1) / 2, 3.70, "相違は\n評価のみ", ha="center", va="bottom",
        fontsize=12, fontweight="bold", linespacing=1.25)

# 下段の対比
ax.text(XS[0] + BW / 2 - 0.40, 0.52, "例：最短経路長・消費電力\n→ 最大化・最小化が可能",
        ha="center", va="top", fontsize=10.5, color=S.INK_2, linespacing=1.3)
ax.text(XS[1] + BW / 2 - 0.40, 0.52,
        "「好みの音」に $f$ は定義できない\n→ 人が代わりに評価",
        ha="center", va="top", fontsize=10.5, color=S.BLUE, linespacing=1.3,
        fontweight="bold")

fig.savefig("local/fig_iec.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_iec.png")
