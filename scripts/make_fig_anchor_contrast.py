#!/usr/bin/env python3
"""S5: アンカーの性質の対比図を生成する。

IteraTTA の audio prior は実音声を順拡散した SDEdit 型の制約であり、音色を保持する
一方でその音の意味的内容も抱えている。本研究が固定するガウスノイズは条件ベクトルと
独立に引かれるため、意味方向への引力を持たない。

対照（IteraTTA）を ORANGE、提案を BLUE とする。

出力: local/fig_anchor_contrast.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 6.75)
ax.axis("off")


def box(x, y, w, h, text, ec=S.INK, fc="white", ls="-", lw=1.4, fs=11.5,
        bold=False, tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.3, color=tc or S.INK,
            fontweight="bold" if bold else "normal")


def arrow(p0, p1, text=None, lw=1.5, ty=0.20, fs=10, color=S.INK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + ty, text,
                ha="center", va="bottom", fontsize=fs, color=color)


# ---- 行の背景（注記のはみ出しを避けるため下端を広めに取る） ----
ax.add_patch(FancyBboxPatch((0.10, 2.72), 12.2, 2.43,
                            boxstyle="round,pad=0.05", linewidth=0,
                            facecolor=S.tint(S.ORANGE, 0.94)))
ax.add_patch(FancyBboxPatch((0.10, 0.12), 12.2, 2.43,
                            boxstyle="round,pad=0.05", linewidth=0,
                            facecolor=S.tint(S.BLUE, 0.93)))

# ---- アンカーの定義（断りなく使わない） ----
ax.add_patch(FancyBboxPatch(
    (0.10, 5.72), 12.2, 0.78,
    boxstyle="round,pad=0.05,rounding_size=0.10",
    linewidth=1.6, edgecolor=S.INK, facecolor=S.PANEL))
ax.text(6.20, 6.11,
        "アンカー ＝ 生成の出発点．これを固定したまま意味のみを変化させたい",
        ha="center", va="center", fontsize=13, fontweight="bold", color=S.INK)

# ---- 列見出し ----
ax.text(3.55, 5.32, "アンカーの作り方", ha="center", va="center", fontsize=12.5,
        fontweight="bold")
ax.text(9.75, 5.32, "意味を大きく動かすと", ha="center", va="center", fontsize=12.5,
        fontweight="bold")
ax.plot([7.05, 7.05], [0.07, 5.18], ls="--", lw=1.0, color=S.AXIS)

# ================= 上段: IteraTTA の audio prior（対照＝ORANGE） =========
ax.text(0.35, 4.80, "IteraTTA の audio prior（SDEdit 型）", ha="left",
        va="center", fontsize=12, fontweight="bold", color=S.ORANGE)

box(0.45, 3.45, 2.30, 0.95, "選んだ生成音\n（実音声）", ec=S.ORANGE)
arrow((2.85, 3.92), (4.05, 3.92), "順拡散", color=S.ORANGE)
box(4.15, 3.45, 2.30, 0.95, "出発点\n$z_{t_0}$", ec=S.ORANGE,
    fc=S.tint(S.ORANGE, 0.72), lw=2.0)
ax.text(5.30, 3.22, "音色を保持．ただし\n意味的特徴も内包", ha="center", va="top",
        fontsize=10, color=S.INK_2, linespacing=1.25)

box(7.45, 3.45, 4.60, 0.95,
    "ノイズ量の増加が必要\n→ アンカー自体が失われる", ec=S.ORANGE,
    fc=S.tint(S.ORANGE, 0.88), lw=1.8)
ax.text(9.75, 3.22, "保持力と探索範囲がトレードオフ", ha="center", va="top",
        fontsize=10, color=S.INK_2)

# ================= 下段: 本研究（提案＝BLUE） =================
ax.text(0.35, 2.20, "本研究：$x_T$ を固定", ha="left", va="center",
        fontsize=12, fontweight="bold", color=S.BLUE)

box(0.45, 0.85, 2.30, 0.95, "ガウスノイズ\n$x_T \\sim \\mathcal{N}(0, I)$",
    ec=S.BLUE)
arrow((2.85, 1.32), (4.05, 1.32), "そのまま", color=S.BLUE)
box(4.15, 0.85, 2.30, 0.95, "出発点\n$x_T^{\\star}$", ec=S.BLUE,
    fc=S.tint(S.BLUE, 0.72), lw=2.0)
ax.text(5.30, 0.62, "$c$ と独立にサンプル\n＝ 意味的に中立", ha="center", va="top",
        fontsize=10, color=S.INK_2, linespacing=1.25)

box(7.45, 0.85, 4.60, 0.95,
    "アンカーが抵抗しない\n→ 意味的に遠方まで探索可能", ec=S.BLUE,
    fc=S.tint(S.BLUE, 0.85), lw=2.0, bold=True, tc=S.BLUE)
ax.text(9.75, 0.62, "集団を広く散らす IEC の要件", ha="center", va="top",
        fontsize=10, color=S.INK_2)

fig.savefig("local/fig_anchor_contrast.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_anchor_contrast.png")
