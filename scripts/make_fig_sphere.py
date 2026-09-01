#!/usr/bin/env python3
"""S9 / S10: CLAP 超球面上の操作を示す2枚の概念図をまとめて生成する。

S9 (fig_slerp.png)          : 交叉（親2体の測地線補間）と Micro-Slerp 変異
S10 (fig_init_population.png): 初期個体群の生成（基準ベクトルから語彙方向へ alpha=0.4）

球は正射影（z 軸方向から見る）で描き、測地線は 3次元で slerp してから射影するので、
弧の形は幾何的に正しい。

配色は意味軸を扱う図なので BLUE を基調とし、語彙プール由来の方向（変異・注入の
向かう先）を ORANGE で示す。

出力: local/fig_slerp.png, local/fig_init_population.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

import figstyle as S

S.use(12)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def slerp3(a, b, t):
    """3次元単位ベクトル間の球面線形補間。t はスカラまたは配列。"""
    a, b = unit(a), unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-6:
        return np.outer(np.atleast_1d(t) * 0 + 1, a).squeeze()
    t = np.atleast_1d(t)[:, None]
    out = (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)
    return out.squeeze()


def proj(v):
    """正射影（x, y をそのまま平面座標に使う）。"""
    v = np.atleast_2d(v)
    return v[:, 0], v[:, 1]


def draw_sphere(ax):
    """球の輪郭と、奥行きの手がかりになる補助線を描く。"""
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.8, edgecolor=S.INK_2,
                        zorder=1))
    ax.add_patch(Circle((0, 0), 1.0, facecolor=S.tint(S.BLUE, 0.96),
                        edgecolor="none", zorder=0))
    for h in (0.42, 0.80):
        ax.add_patch(Ellipse((0, 0), 2.0, 2.0 * h, fill=False, lw=0.9,
                             ls=":", edgecolor=S.AXIS, zorder=1))
        ax.add_patch(Ellipse((0, 0), 2.0 * h, 2.0, fill=False, lw=0.9,
                             ls=":", edgecolor=S.AXIS, zorder=1))


def geodesic(ax, a, b, **kw):
    """a から b への測地線（大円の弧）を描く。"""
    pts = slerp3(a, b, np.linspace(0, 1, 120))
    xs, ys = proj(pts)
    ax.plot(xs, ys, zorder=3, **kw)


def dot(ax, v, label, dx=0.0, dy=0.0, filled=True, ms=11, fs=12,
        ha="center", va="center", bold=False, color=None):
    col = color or S.BLUE
    x, y = proj(v)
    ax.plot(x, y, "o", ms=ms, mfc=col if filled else "white",
            mec=col, mew=1.6, zorder=5)
    if label:
        ax.text(float(x) + dx, float(y) + dy, label, ha=ha, va=va,
                fontsize=fs, zorder=6, color=col if bold else S.INK,
                fontweight="bold" if bold else "normal")


# 注記を球に重ねないよう、左側に余白を確保したキャンバスを使う
XMIN, XMAX = -2.10, 1.42
YMIN, YMAX = -1.62, 1.34
XMID = (XMIN + XMAX) / 2


def finish(ax, title, note, note_h=0.30):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(XMID, 1.24, title, ha="center", va="center", fontsize=13.5,
            fontweight="bold")
    ax.add_patch(FancyBboxPatch(
        (XMIN + 0.05, YMIN + 0.04), (XMAX - XMIN) - 0.10, note_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor=S.BLUE, facecolor=S.tint(S.BLUE, 0.90),
        zorder=2))
    ax.text(XMID, YMIN + 0.04 + note_h / 2, note, ha="center", va="center",
            fontsize=11, linespacing=1.3, zorder=3, color=S.INK)


# =====================================================================
# S9: 交叉と Micro-Slerp 変異
# =====================================================================
fig, ax = plt.subplots(figsize=(6.4, 5.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
draw_sphere(ax)

p1 = unit([-0.62, 0.30, 0.72])
p2 = unit([0.66, 0.12, 0.74])
pool = unit([0.10, -0.78, 0.62])
child = slerp3(p1, p2, 0.45)
mutated = slerp3(child, pool, 0.22)

geodesic(ax, p1, p2, color=S.BLUE, lw=2.2)
geodesic(ax, child, pool, color=S.ORANGE, lw=1.8, ls="--")
# 変異が向かう先（語彙方向）は参考として示す
dot(ax, pool, "$c_{pool}$", dy=-0.13, filled=False, ms=9, fs=11,
    color=S.ORANGE)

dot(ax, p1, "$c_{p_1}$", dx=-0.05, dy=0.15, fs=12.5)
dot(ax, p2, "$c_{p_2}$", dx=0.05, dy=0.15, fs=12.5)
dot(ax, child, "子", dx=-0.17, dy=0.05, filled=False, fs=12)
dot(ax, mutated, "変異後", dx=0.20, dy=0.02, ms=12, fs=12, bold=True,
    color=S.ORANGE)

ax.annotate("交叉\n$\\alpha \\sim \\mathcal{U}(0.3,\\,0.7)$",
            xy=tuple(float(v) for v in proj(slerp3(p1, p2, 0.20))),
            xytext=(XMIN + 0.06, 0.78), fontsize=11, ha="left", va="center",
            linespacing=1.3, color=S.BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=S.BLUE,
                            connectionstyle="arc3,rad=0.15"))
ax.annotate("Micro-Slerp 変異\n$\\mu \\sim \\mathcal{U}(0.10,\\,0.25)$",
            xy=tuple(float(v) for v in proj(slerp3(child, pool, 0.12))),
            xytext=(XMIN + 0.06, -0.62), fontsize=11, ha="left", va="center",
            linespacing=1.3, color=S.ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=S.ORANGE,
                            connectionstyle="arc3,rad=-0.15"))

finish(ax, "CLAP 超球面上の遺伝的操作",
       "すべての操作を球面線形補間で行い，多様体上に留める")
fig.savefig("local/fig_slerp.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_slerp.png")
plt.close(fig)

# =====================================================================
# S10: 初期個体群の生成（＝多様性注入と同じ手続き）
# =====================================================================
fig, ax = plt.subplots(figsize=(6.4, 5.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
draw_sphere(ax)

base = unit([-0.05, 0.30, 1.00])
dirs = [
    unit([-0.88, 0.10, 0.46]),
    unit([-0.45, -0.72, 0.53]),
    unit([0.22, -0.90, 0.38]),
    unit([0.80, -0.30, 0.52]),
    unit([0.86, 0.36, 0.36]),
]
for i, d in enumerate(dirs):
    geodesic(ax, base, d, color=S.AXIS, lw=1.1, ls=":")
    c_i = slerp3(base, d, 0.4)
    x, y = proj(c_i)
    ax.plot(x, y, "o", ms=10, mfc="white", mec=S.BLUE, mew=1.8, zorder=5)
    xd, yd = proj(d)
    ax.plot(xd, yd, "x", ms=8, mew=1.8, color=S.ORANGE, zorder=4)

dot(ax, base, "$c_{base}$", dx=0.0, dy=0.17, ms=13, fs=13, bold=True)
ax.annotate("$\\alpha = 0.4$ の点が初期個体",
            xy=tuple(float(v) for v in proj(slerp3(base, dirs[0], 0.4))),
            xytext=(XMIN + 0.06, 0.62), fontsize=11.5, ha="left", va="center",
            color=S.BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=S.BLUE,
                            connectionstyle="arc3,rad=0.22"))
ax.annotate("×：語彙の埋め込み $c_{rand}$",
            xy=tuple(float(v) for v in proj(dirs[1])),
            xytext=(XMIN + 0.06, -0.70), fontsize=11, ha="left", va="center",
            color=S.ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=S.ORANGE,
                            connectionstyle="arc3,rad=-0.15"))

finish(ax, "初期個体群の生成",
       "多様性注入も同一の手続き\n→ 語彙が整合しないと最初の世代から逸脱",
       note_h=0.48)
fig.savefig("local/fig_init_population.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_init_population.png")
plt.close(fig)
