#!/usr/bin/env python3
"""S5.1: 探索パラメータの決定（修正版実験）の結果を2枚のパネルで示す。

データは scripts/outputs/exp_search_params/runs.csv（各条件 目標3種 × 10試行 = 30試行）。
旧 exp1 / exp3 の1試行データは使わない。

左: 初期個体群の作り方（3方式）… 変異率は 0.4 に固定
右: 変異率（3水準）＋「変異も注入も無し」… 初期個体群は条件間で同一

配色の使い分け:
  左は「3つの異なる方式」＝カテゴリカルなので3スロットの色相。
  右は「1つの数値の3水準」＝順序尺度なので青の濃淡ランプ。対照条件のみ ORANGE。
  符号化を変えることで、左右の系列に対応関係があるという誤解を防ぐ。

帯は平均±標準偏差。旧版と違い各条件30試行なので、ばらつきを必ず示す。

出力: local/fig_exp_curves.png (300 dpi)
"""
import csv
import os
import statistics
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as S

S.use(12)

CSV = "scripts/outputs/exp_search_params/runs.csv"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]      # 順序尺度用（薄い＝小さい値）

PANELS = [
    ("A_init_method", "① 初期個体群の作り方", "変異率は $p_{mut}=0.4$ に固定",
     "lower right", [
         ("slerp_prompt", "語彙の埋め込みとの Slerp", S.BLUE, "-", "o"),
         ("slerp_random", "超球面上の一様サンプル", S.ORANGE, "--", "s"),
         ("gaussian", "ガウス変異", S.AQUA, "-.", "^"),
     ]),
    ("B_mutation_rate", "② 変異率", "初期個体群は条件間で同一",
     "lower right", [
         ("p_mut=0.5", "$p_{mut}=0.5$", RAMP[2], "-", "o"),
         ("p_mut=0.3", "$p_{mut}=0.3$", RAMP[1], "--", "s"),
         ("p_mut=0.0", "$p_{mut}=0.0$", RAMP[0], ":", "x"),
         ("p_mut=0.0,注入なし", "変異・注入とも無効", S.ORANGE, "-", "D"),
     ]),
]


def load():
    """(comparison, condition) -> 世代ごとの値リスト を返す。"""
    d = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            d[(r["comparison"], r["condition"])][int(r["generation"])].append(
                float(r["max_fitness"]))
    return d


data = load()
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.20, wspace=0.20)

n_trials = None
for ax, (comp, title, subtitle, loc, spec) in zip(axes, PANELS):
    for cond, label, color, ls, marker in spec:
        per_gen = data[(comp, cond)]
        gens = sorted(per_gen)
        mean = [statistics.mean(per_gen[g]) for g in gens]
        sd = [statistics.pstdev(per_gen[g]) for g in gens]
        n_trials = len(per_gen[gens[0]])
        ax.fill_between(gens, [m - s for m, s in zip(mean, sd)],
                        [m + s for m, s in zip(mean, sd)],
                        color=color, alpha=0.13, linewidth=0)
        ax.plot(gens, mean, ls=ls, marker=marker, ms=6.5, lw=2.2, color=color,
                label=label, mfc="white", mew=1.6)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=22, loc="left",
                 color=S.INK)
    ax.text(0.0, 1.035, subtitle, transform=ax.transAxes, fontsize=10.5,
            color=S.INK_2, va="bottom")
    ax.set_xlabel("世代", fontsize=11.5)
    ax.set_ylabel("最大適合度（代理）", fontsize=11.5)
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(0, 1.0)
    ax.grid(True, ls=":", lw=0.8, color=S.GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(S.AXIS)
    ax.legend(fontsize=9.5, loc=loc, framealpha=1.0, edgecolor=S.AXIS,
              labelcolor=S.INK)
    ax.tick_params(labelsize=10, color=S.AXIS)

# 対照条件が完全に停止していることを強調する
ax = axes[1]
ax.annotate("変異と注入を両方無効にした条件のみ\n多様性を失い停滞",
            xy=(3.4, 0.463), xytext=(0.35, 0.17), fontsize=10.5,
            ha="left", va="center", linespacing=1.35, color=S.ORANGE,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=S.ORANGE,
                            connectionstyle="arc3,rad=0.25"))

fig.text(0.5, 0.045,
         "帯は平均 ± 標準偏差．各条件 目標プロンプト3種 × %d試行 = %d試行．"
         "2つの比較は同一の目標プロンプトを用いている．" % (10, n_trials),
         ha="center", va="center", fontsize=10, color=S.MUTED)

fig.savefig("local/fig_exp_curves.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote local/fig_exp_curves.png")
