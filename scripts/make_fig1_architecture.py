#!/usr/bin/env python3
"""図1: 提案手法アーキテクチャ（二段階交互探索）のブロック図を生成する。

冊子体は白黒印刷のため、グレースケール・実線/破線で区別する。

出力: local/fig1_architecture.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk)
_jp = fm.FontProperties(fname=_cjk).get_name()
plt.rcParams.update({
    "font.size": 13, "font.family": _jp, "axes.unicode_minus": False,
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ls="-", lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw, edgecolor="black", facecolor=fc, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=13, linespacing=1.25)


def arrow(p0, p1, ls="-", text=None, rad=0.0, tx=0, ty=0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
        color="black", linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + tx, (p0[1] + p1[1]) / 2 + ty
        ax.text(mx, my, text, ha="center", va="center", fontsize=11)


# ---- Phase panels (background) ----
ax.add_patch(FancyBboxPatch((0.15, 0.3), 4.6, 4.0,
             boxstyle="round,pad=0.05", linewidth=1.0, edgecolor="black",
             facecolor="0.95", linestyle="-"))
ax.add_patch(FancyBboxPatch((5.25, 0.3), 4.6, 4.0,
             boxstyle="round,pad=0.05", linewidth=1.0, edgecolor="black",
             facecolor="0.90", linestyle="-"))
ax.text(2.45, 4.05, "音響テクスチャ選択", ha="center",
        fontsize=14, fontweight="bold")
ax.text(7.55, 4.05, "意味空間進化", ha="center",
        fontsize=14, fontweight="bold")

# ---- Phase 1 blocks ----
box(0.5, 3.0, 3.9, 0.7, "プロンプト $y$ $\\rightarrow$ $c_{base}=$CLAP$(y)$")
box(0.5, 1.9, 3.9, 0.8,
    "候補を $N$ 個生成\n（$c_{base}$ 共通・$x_T$ のみ変える）")
box(0.5, 0.7, 3.9, 0.8, "1つ選択\n$\\Rightarrow$ 音響テクスチャ $x_T^{\\star}$ を固定")
arrow((2.45, 3.0), (2.45, 2.72))
arrow((2.45, 1.9), (2.45, 1.52))

# ---- Phase 2 blocks ----
box(5.6, 3.0, 3.9, 0.8,
    "次世代: エリート / Slerp交叉\n/ 微小Slerp変異 / 注入")
box(5.6, 1.9, 3.9, 0.8, "2個以上を選択（淘汰）")
box(5.6, 0.7, 3.9, 0.8,
    "CLAP超球面上の個体群\n（$x_T^{\\star}$ 固定・$c$ を変える）")
arrow((7.55, 2.72), (7.55, 3.0))
arrow((7.55, 1.52), (7.55, 1.9))
# loop back inside phase2 (next gen -> population)
# ax.add_patch(FancyArrowPatch((9.65, 1.1), (9.65, 3.4),
#              arrowstyle="-|>", linewidth=2.0, color="black"))
# ax.add_patch(FancyArrowPatch((9.65, 1.1), (9.5, 1.1),
#              arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="black"))
arrow((9.65, 3.4), (9.65, 1.1))
ax.text(9.92, 2.25, "反復", rotation=90, va="center", fontsize=11)

# ---- inter-phase arrows ----
arrow((4.4, 1.1), (5.6, 1.1), text="$x_T^{\\star}$ を固定", ty=0.28)
arrow((5.6, 3.4), (4.4, 3.4), ls="--", text="$c^{\\star}$ を継承", ty=0.28)

fig.tight_layout()
out = "local/fig1_architecture.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved: {out}")
