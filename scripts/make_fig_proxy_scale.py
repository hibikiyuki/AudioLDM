#!/usr/bin/env python3
"""質疑用: 代理適合度 0.89 が「どれくらい近いのか」を示す数直線。

参照点はすべて修正版実験（scripts/outputs/exp_search_params/runs.csv）の実測値。
各条件 目標プロンプト3種 × 10試行 = 30試行の平均。
- 0.12 : 初期集団の平均（ガウス摂動＝目標と無関係な方向）
- 0.17 : 同条件の初期最良個体
- 0.47 : 語彙方向へ Slerp した初期集団の最良個体
- 0.69 : 同条件で8世代後の最良個体

絶対値の解釈は難しく、条件間の相対比較として読むべきであることも図中に明記する。

出発点側を ORANGE、到達点を BLUE、参照用の無関係な水準を MUTED で示す。

出力: local/fig_proxy_scale.png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S

S.use(12)

fig, ax = plt.subplots(figsize=(9.2, 3.5))
ax.set_xlim(-0.05, 1.08)
ax.set_ylim(-0.78, 1.05)
ax.axis("off")

Y = 0.0

# ---- 数直線 ----
ax.add_patch(FancyArrowPatch((0, Y), (1.05, Y), arrowstyle="-|>",
                             mutation_scale=13, linewidth=1.5, color=S.INK))
for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    ax.plot([v, v], [Y - 0.035, Y + 0.035], lw=1.2, color=S.INK_2)
    ax.text(v, Y - 0.10, f"{v:.1f}", ha="center", va="top", fontsize=10.5,
            color=S.MUTED)

# (値, ラベル, ラベルのy, 色, 塗るか)
MARKS = [
    (0.116, "初期集団の平均\n（目標と無関係な方向）", 0.36, S.MUTED, False),
    (0.166, "その最良個体", 0.78, S.MUTED, False),
    (0.468, "語彙方向へ振った\n初手の最良個体", 0.36, S.ORANGE, True),
    (0.694, "8世代後の最良個体", 0.78, S.BLUE, True),
]

for v, label, ly, col, filled in MARKS:
    ax.plot(v, Y, "o", ms=14, mfc=col if filled else "white", mec=col,
            mew=1.8, zorder=4)
    ax.plot([v, v], [Y + 0.05, ly - 0.055], ls=":", lw=1.1, color=col,
            zorder=1)
    ax.text(v, ly, label, ha="center", va="bottom", fontsize=10.5,
            linespacing=1.25, color=S.INK)
    ax.text(v, Y + 0.055, f"{v:.2f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold", zorder=5, color=col,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

# ---- 探索で動いた区間 ----
ax.add_patch(FancyArrowPatch((0.468, Y - 0.24), (0.694, Y - 0.24),
                             arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.8, color=S.BLUE))
ax.text(0.58, Y - 0.30, "8世代の探索による変化量", ha="center", va="top",
        fontsize=10.5, color=S.BLUE, fontweight="bold")

# ---- 但し書き（2行に折り、枠からはみ出さない幅に収める） ----
ax.add_patch(FancyBboxPatch(
    (0.00, -0.74), 1.02, 0.26,
    boxstyle="round,pad=0.012,rounding_size=0.02",
    linewidth=1.1, edgecolor=S.AXIS, facecolor=S.PANEL, linestyle="--"))
ax.text(0.51, -0.61,
        "※ CLAP テキスト埋め込み同士のコサイン類似度（1.0 が完全一致）。\n"
        "絶対値の解釈は困難であり，条件間の相対比較として読む",
        ha="center", va="center", fontsize=10, color=S.INK_2,
        linespacing=1.35)

ax.set_title("代理適合度のスケール（各条件30試行の平均）",
             fontsize=12.5, fontweight="bold", pad=14, loc="left", x=-0.03)

fig.savefig("local/fig_proxy_scale.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_proxy_scale.png")
