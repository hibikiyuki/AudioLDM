#!/usr/bin/env python3
"""S17: アンカー設計の比較実験で何を測るかを示す模式図を生成する。

⚠️ 本図は実測値ではなく仮説の図示である。SDEdit 型アンカーが保持と移動距離の
トレードオフ曲線を描くのに対し、提案手法（x_T 固定）がその外側に出るかどうかを
問う、という実験の構図を示す。図中にも「予想」「未検証」を明記する。

対照（SDEdit 型）を ORANGE、提案を BLUE とする。

出力: local/fig_anchor_tradeoff.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Ellipse

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(7.0, 4.4))
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.16)

# ---- SDEdit 型アンカーのトレードオフ曲線（予想） ----
noise = np.array([0.2, 0.4, 0.6, 0.8])
move = np.array([0.18, 0.38, 0.60, 0.82])      # 意味的な移動距離
keep = np.array([0.90, 0.72, 0.46, 0.18])      # テクスチャの保持

t = np.linspace(0, 1, 200)
curve_x = np.interp(t, np.linspace(0, 1, 4), move)
curve_y = np.interp(t, np.linspace(0, 1, 4), keep)
ax.plot(curve_x, curve_y, "-", lw=2.2, color=S.ORANGE, zorder=2)
ax.plot(move, keep, "s", ms=9, mfc="white", mec=S.ORANGE, mew=1.8, zorder=3)
for m, k, n in zip(move, keep, noise):
    ax.annotate(f"{n:.1f}", (m, k), textcoords="offset points",
                xytext=(9, 7), fontsize=9.5, color=S.ORANGE)
ax.text(0.05, 0.46, "SDEdit 型アンカー\n（ノイズ量を掃引）",
        fontsize=11, color=S.ORANGE, linespacing=1.3, ha="left", va="top",
        fontweight="bold")

# ---- 提案手法（x_T 固定）の予想位置 ----
px, py = 0.80, 0.80
ax.add_patch(Ellipse((px, py), 0.20, 0.20, fill=False, ls="--", lw=1.5,
                     edgecolor=S.BLUE, zorder=2))
ax.plot(px, py, "*", ms=24, mfc=S.BLUE, mec=S.BLUE, mew=1.0, zorder=4)
ax.annotate("提案手法（$x_T$ 固定）\n【予想・未検証】",
            xy=(px, py), xytext=(0.30, 0.93), fontsize=11.5,
            fontweight="bold", linespacing=1.3, va="top", color=S.BLUE,
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=S.BLUE,
                            connectionstyle="arc3,rad=-0.25"))

# ---- 争点の注記 ----
ax.annotate("", xy=(px - 0.02, py - 0.13), xytext=(0.64, 0.44),
            arrowprops=dict(arrowstyle="<->", lw=1.3, color=S.INK_2,
                            linestyle="--"))
ax.text(0.845, 0.44, "曲線の外側に\n位置するか", fontsize=10.5, color=S.INK,
        ha="center", va="top", linespacing=1.3)

# ---- 軸 ----
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.set_xlabel("意味的な移動距離（起点からの CLAP 余弦距離）", fontsize=11.5)
ax.set_ylabel("テクスチャの保持（MFCC 距離の小ささ）", fontsize=10.5)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.tick_params(labelsize=10, color=S.AXIS)
ax.grid(True, ls=":", lw=0.8, color=S.GRID)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax.spines[side].set_color(S.AXIS)

ax.set_title("アンカー設計の比較：何を測るか", fontsize=13, fontweight="bold",
             pad=12)

# ---- 但し書き ----
ax.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.52, 0.10, boxstyle="round,pad=0.012,rounding_size=0.02",
    linewidth=1.0, edgecolor=S.AXIS, facecolor=S.PANEL, linestyle="--",
    transform=ax.transData, zorder=5))
ax.text(0.28, 0.07, "※ 本図は仮説の図示であり，実測値ではない",
        ha="center", va="center", fontsize=9.5, color=S.INK_2, zorder=6)

fig.savefig("local/fig_anchor_tradeoff.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_anchor_tradeoff.png")
