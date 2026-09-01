#!/usr/bin/env python3
"""S2: 拡散モデルの2入力（条件ベクトルと初期ノイズ）の役割を示すブロック図。

条件ベクトル c が意味的方向を、初期ノイズ x_T が音響テクスチャを支配する、という
本研究の出発点を1枚で示す。

意味軸を BLUE、テクスチャ軸を ORANGE とし、この対応は発表全体で固定する。

出力: local/fig_two_inputs.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(8.6, 3.5))
ax.set_xlim(0, 13.0)
ax.set_ylim(0, 5.0)
ax.axis("off")


def box(x, y, w, h, text, ec=S.INK, fc="white", lw=1.4, fs=12, bold=False,
        tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.3, color=tc or S.INK,
            fontweight="bold" if bold else "normal")


def arrow(p0, p1, rad=0.0, lw=1.5, color=S.INK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}"))


# ---- 上段：意味の経路（BLUE） ----
box(0.25, 3.35, 2.20, 1.00, "プロンプト\n$y$", ec=S.BLUE,
    fc=S.tint(S.BLUE, 0.92))
arrow((2.57, 3.85), (3.15, 3.85), color=S.BLUE)
box(3.25, 3.35, 3.10, 1.00, "CLAP\nテキストエンコーダ", ec=S.BLUE,
    fc=S.tint(S.BLUE, 0.92), fs=11.5)
arrow((6.47, 3.85), (7.15, 3.85), color=S.BLUE)
box(7.25, 3.35, 1.85, 1.00, "$c$", ec=S.BLUE, fc=S.tint(S.BLUE, 0.72),
    lw=2.0, fs=15, bold=True)
ax.text(8.18, 3.15, "意味的方向\nジャンル・情動・楽器編成", ha="center", va="top",
        fontsize=10.5, color=S.BLUE, linespacing=1.3, fontweight="bold")

# ---- 下段：テクスチャの経路（ORANGE・上段と右端を揃える） ----
box(0.25, 0.55, 6.10, 1.00, "ガウスノイズ  $x_T \\sim \\mathcal{N}(0,\\,I)$",
    ec=S.ORANGE, fc=S.tint(S.ORANGE, 0.92), fs=13)
arrow((6.47, 1.05), (7.15, 1.05), color=S.ORANGE)
box(7.25, 0.55, 1.85, 1.00, "$x_T$", ec=S.ORANGE, fc=S.tint(S.ORANGE, 0.72),
    lw=2.0, fs=15, bold=True)
ax.text(8.18, 0.35, "音響テクスチャ\n音色・空間性・録音の質感", ha="center", va="top",
        fontsize=10.5, color=S.ORANGE, linespacing=1.3, fontweight="bold")

# ---- サンプラへ合流 ----
arrow((9.20, 3.85), (10.55, 2.85), rad=-0.18, color=S.BLUE)
arrow((9.20, 1.05), (10.55, 2.05), rad=0.18, color=S.ORANGE)
box(10.65, 1.95, 2.10, 1.00, "サンプラ\n$\\mathcal{G}$", fs=13, fc=S.PANEL)
ax.text(11.70, 3.35, "$\\hat a = \\mathcal{G}(c,\\ x_T)$", ha="center",
        va="center", fontsize=13)
arrow((11.70, 1.90), (11.70, 1.15))
ax.text(11.70, 0.92, "音響信号", ha="center", va="top", fontsize=12,
        fontweight="bold")

fig.savefig("local/fig_two_inputs.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_two_inputs.png")
