#!/usr/bin/env python3
"""探索パラメータの決定（修正版）。

旧 exp1 / exp3 には以下の欠陥があり、質疑で指摘された。本スクリプトはそれらを
すべて解消したうえで、2つの比較を同一の枠組みで実行する。旧スクリプトと旧結果は
scripts/exp{1,3}_*.py および scripts/outputs/exp{1,3}/ にそのまま残してある。

  旧の欠陥                                   本スクリプトでの対処
  ------------------------------------------ --------------------------------
  1) 各条件1試行のみ                          N_TRIALS 試行の平均±標準偏差
  2) 変異率比較でも初期個体群を条件ごとに      試行ごとに初期個体群を1つ作り、
     独立サンプル → 第0世代に差が出ていた      3条件で完全に同一のものを使う
  3) 2つの比較で目標プロンプトが異なり、       全比較で同一の目標プロンプト集合
     結果を並べられなかった                    （複数）を使い、平均する
  4) 初期化方式の比較で slerp_prompt だけ      3条件すべて同じシミュレーション
     実パイプライン、他はシミュレーション      経路で生成する
  5) 語彙プールが実行ごとに変わりうる          固定シードで一度だけサンプルし、
                                             全条件・全試行で共有する

音声生成は行わない。必要なのは CLAP テキスト埋め込みだけで、探索ループは
512次元ベクトルの演算のみである。

使用法:
    NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
      python scripts/exp_search_params.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.exp_common import (
    compute_proxy_fitness, select_top_k, pairwise_cosine_distance,
    centroid_distance,
    simulate_gaussian_population, simulate_slerp_random_population,
    simulate_slerp_prompt_population, simulate_next_generation,
)

OUTPUT_DIR = "scripts/outputs/exp_search_params"

POP_SIZE = 6
N_GENERATIONS = 8
SELECT_K = 2
N_TRIALS = 10
POOL_SIZE = 30
POOL_SEED = 20260822          # 語彙プールのサンプリングを固定する
BASE_PROMPT = "music"
STAGNANT_EPS = 0.01

TARGETS = [
    "tense orchestral strings with brass fanfare",
    "energetic electronic dance music with heavy bass",
    "gentle acoustic guitar ballad",
]

# 比較A: 初期個体群の作り方（変異率は下記 P_MUT_FIXED に固定）
P_MUT_FIXED = 0.4
INIT_METHODS = ["slerp_prompt", "slerp_random", "gaussian"]
INIT_PARAM = {"slerp_prompt": 0.3, "slerp_random": 0.3, "gaussian": 0.1}

# 比較B: 変異率（初期個体群は slerp_prompt に固定し、条件間で同一のものを使う）
INIT_FIXED = "slerp_prompt"
P_MUT_VALUES = [0.0, 0.3, 0.5]
# 追加条件: 変異も注入も切る（新しい方向が一切入らない）
NO_NEW_DIRECTION = "p_mut=0.0,注入なし"


def seed_all(seed: int) -> None:
    """simulate_* が使う random / np.random を両方固定する。"""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def make_population(method: str, base_emb, pool_embs):
    p = INIT_PARAM[method]
    if method == "gaussian":
        return simulate_gaussian_population(base_emb, p, POP_SIZE)
    if method == "slerp_random":
        return simulate_slerp_random_population(base_emb, p, POP_SIZE)
    if method == "slerp_prompt":
        return simulate_slerp_prompt_population(base_emb, pool_embs, p, POP_SIZE)
    raise ValueError(method)


def run_one(population, target_emb, pool_embs, p_mut, n_inject=1,
            elite=2, mu_range=(0.05, 0.15), selection="top_k"):
    """1試行を回し、世代ごとの記録を返す。population は破壊しない。

    selection:
      "top_k"  - 代理適合度の上位 SELECT_K 体を選択（通常の探索）
      "random" - 適合度を無視してランダムに SELECT_K 体を選択（**対照条件**）

    "random" は「世代ごとに適合度が上がるのは選択が効いているからか、
    それとも Slerp の幾何とエリート保存で自動的にそうなるだけか」を切り分けるための対照。
    ランダム選択でも同じだけ上昇するなら、上昇は探索の成果ではない。
    """
    pop = [t.clone() for t in population]
    logs, prev_sel, streak = [], None, 0
    for gen in range(N_GENERATIONS):
        fit = compute_proxy_fitness(pop, target_emb)
        if selection == "random":
            sel = list(np.random.choice(len(pop), size=SELECT_K, replace=False))
        else:
            sel = select_top_k(fit, k=SELECT_K)
        sel_embs = [pop[i] for i in sel]
        cdist = centroid_distance(prev_sel, sel_embs) if prev_sel else -1.0
        streak = streak + 1 if (prev_sel and 0 <= cdist < STAGNANT_EPS) else 0
        prev_sel = sel_embs
        logs.append({
            "generation": gen,
            "max_fitness": float(max(fit)),
            "mean_fitness": float(np.mean(fit)),
            "diversity": float(pairwise_cosine_distance(pop)),
            "stagnant_streak": streak,
        })
        if gen < N_GENERATIONS - 1:
            pop = simulate_next_generation(
                pop, sel, pool_embs,
                elite_count=elite, random_sample_count=n_inject, p_mut=p_mut,
                mu_range=mu_range,
            )
    return logs


def main() -> None:
    global P_MUT_FIXED, INIT_PARAM, OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="audioldm-m-full")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--match-impl", action="store_true",
                    help="デモUI/パイプラインの既定値に合わせる"
                         "（α=0.4, μ=(0.10,0.25), エリート1, 比較Aの p_mut=0.5）")
    ap.add_argument("--out", default=OUTPUT_DIR)
    args = ap.parse_args()

    ELITE, MU = 2, (0.05, 0.15)
    if args.match_impl:
        INIT_PARAM = {"slerp_prompt": 0.4, "slerp_random": 0.4, "gaussian": 0.1}
        P_MUT_FIXED, ELITE, MU = 0.5, 1, (0.10, 0.25)
        print("[--match-impl] α=0.4, p_mut(比較A)=0.5, エリート=1, μ=(0.10,0.25)")
    OUTPUT_DIR = args.out

    from audioldm.iec_pipeline import AudioLDM_IEC
    from audioldm.prompt_pool import sample_prompts

    print("モデルをロード中（CLAP テキストエンコーダのみ使用）...")
    pipe = AudioLDM_IEC(model_name=args.model, ga_mode="conditioning",
                        population_size=POP_SIZE)

    base_emb = pipe._encode_text_single(BASE_PROMPT).cpu()
    target_embs = {t: pipe._encode_text_single(t).cpu() for t in TARGETS}

    # 語彙プールは固定シードで一度だけサンプルし、全条件・全試行で共有する
    seed_all(POOL_SEED)
    pool_prompts = sample_prompts(POOL_SIZE, exclude=[BASE_PROMPT] + TARGETS)
    pool_embs = [pipe._encode_text_single(p).cpu() for p in pool_prompts]
    print("語彙プール %d 件を固定（seed=%d）" % (len(pool_embs), POOL_SEED))

    rows = []

    # ---- 比較A: 初期個体群の作り方 -------------------------------------
    print("\n[比較A] 初期個体群の作り方  (p_mut=%.1f 固定)" % P_MUT_FIXED)
    for ti, target in enumerate(TARGETS):
        for trial in range(args.trials):
            for method in INIT_METHODS:
                # 試行ごとにシードを揃え、条件間で乱数の出発点を共有する
                seed_all(1000 + ti * 100 + trial)
                pop = make_population(method, base_emb, pool_embs)
                logs = run_one(pop, target_embs[target], pool_embs, P_MUT_FIXED,
                               elite=ELITE, mu_range=MU)
                for lg in logs:
                    rows.append({"comparison": "A_init_method", "condition": method,
                                 "target": target, "trial": trial, **lg})

    # ---- 比較B: 変異率（初期個体群は条件間で完全に同一） ------------------
    print("[比較B] 変異率  (初期個体群は %s に固定・条件間で同一)" % INIT_FIXED)
    for ti, target in enumerate(TARGETS):
        for trial in range(args.trials):
            # 試行ごとに初期個体群を1つだけ作り、3条件でそれを共有する
            seed_all(2000 + ti * 100 + trial)
            shared_pop = make_population(INIT_FIXED, base_emb, pool_embs)
            for p_mut in P_MUT_VALUES:
                seed_all(2000 + ti * 100 + trial)   # 進化側の乱数も揃える
                logs = run_one(copy.deepcopy(shared_pop), target_embs[target],
                               pool_embs, p_mut, elite=ELITE, mu_range=MU)
                for lg in logs:
                    rows.append({"comparison": "B_mutation_rate",
                                 "condition": "p_mut=%.1f" % p_mut,
                                 "target": target, "trial": trial, **lg})
            # 新しい方向が一切入らない条件（変異も注入も無し）
            seed_all(2000 + ti * 100 + trial)
            logs = run_one(copy.deepcopy(shared_pop), target_embs[target],
                           pool_embs, 0.0, n_inject=0, elite=ELITE, mu_range=MU)
            for lg in logs:
                rows.append({"comparison": "B_mutation_rate",
                             "condition": NO_NEW_DIRECTION,
                             "target": target, "trial": trial, **lg})

    # ---- 比較C: 選択はそもそも効いているか（ランダム選択との対照） --------
    # 「適合度が世代ごとに上がるのは当たり前ではないか」への回答。
    # 上位選択とランダム選択の差が、選択という操作の寄与そのものになる。
    # 差が出なければ、上昇は Slerp の幾何とエリート保存の副産物だったことになる。
    print("[比較C] 選択の寄与  (上位選択 vs ランダム選択・初期個体群は共有)")
    for ti, target in enumerate(TARGETS):
        for trial in range(args.trials):
            seed_all(3000 + ti * 100 + trial)
            shared_pop = make_population(INIT_FIXED, base_emb, pool_embs)
            for sel_mode, label in (("top_k", "上位選択"), ("random", "ランダム選択")):
                seed_all(3000 + ti * 100 + trial)
                logs = run_one(copy.deepcopy(shared_pop), target_embs[target],
                               pool_embs, P_MUT_FIXED, elite=ELITE, mu_range=MU,
                               selection=sel_mode)
                for lg in logs:
                    rows.append({"comparison": "C_selection", "condition": label,
                                 "target": target, "trial": trial, **lg})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "runs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n書き出し: %s (%d 行)" % (csv_path, len(rows)))

    # ---- 集計 -----------------------------------------------------------
    summary = []
    for comp in ["A_init_method", "B_mutation_rate", "C_selection"]:
        if comp == "A_init_method":
            conds = INIT_METHODS
        elif comp == "B_mutation_rate":
            conds = ["p_mut=%.1f" % v for v in P_MUT_VALUES] + [NO_NEW_DIRECTION]
        else:
            conds = ["上位選択", "ランダム選択"]
        print("\n=== %s ===" % comp)
        print("  %-18s %14s %14s %14s %8s %8s" %
              ("条件", "第0世代", "第7世代", "改善量", "最長停滞", "最終多様性"))
        for cond in conds:
            g0, g7, stag, div7 = [], [], [], []
            for target in TARGETS:
                for trial in range(args.trials):
                    sub = [r for r in rows if r["comparison"] == comp
                           and r["condition"] == cond and r["target"] == target
                           and r["trial"] == trial]
                    sub.sort(key=lambda r: r["generation"])
                    g0.append(sub[0]["max_fitness"])
                    g7.append(sub[-1]["max_fitness"])
                    stag.append(max(r["stagnant_streak"] for r in sub))
                    div7.append(sub[-1]["diversity"])
            delta = [b - a for a, b in zip(g0, g7)]
            rec = {
                "comparison": comp, "condition": cond, "n": len(g0),
                "gen0_mean": statistics.mean(g0), "gen0_sd": statistics.pstdev(g0),
                "gen7_mean": statistics.mean(g7), "gen7_sd": statistics.pstdev(g7),
                "delta_mean": statistics.mean(delta), "delta_sd": statistics.pstdev(delta),
                "max_stagnant_mean": statistics.mean(stag),
                "diversity_gen7_mean": statistics.mean(div7),
            }
            summary.append(rec)
            print("  %-18s %6.3f ±%.3f %6.3f ±%.3f %6.3f ±%.3f %8.1f %8.3f" % (
                cond, rec["gen0_mean"], rec["gen0_sd"], rec["gen7_mean"],
                rec["gen7_sd"], rec["delta_mean"], rec["delta_sd"],
                rec["max_stagnant_mean"], rec["diversity_gen7_mean"]))

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({"config": {
            "pop_size": POP_SIZE, "n_generations": N_GENERATIONS,
            "select_k": SELECT_K, "n_trials": args.trials,
            "targets": TARGETS, "base_prompt": BASE_PROMPT,
            "pool_size": POOL_SIZE, "pool_seed": POOL_SEED,
            "p_mut_fixed_in_A": P_MUT_FIXED, "init_fixed_in_B": INIT_FIXED,
        }, "summary": summary}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("\n集計を %s/summary.{json,csv} に保存しました。" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
