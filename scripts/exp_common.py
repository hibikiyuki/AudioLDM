"""共通インフラ: 代理適合度計算、世代ログ記録、結果 CSV 出力。

実験スクリプト (exp1〜exp4) から import して使用する。
モデル不要のユニットテストモードと実モデルを使う統合テストモードを持つ。
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class GenerationLog:
    generation: int
    approach: str
    param_name: str
    param_value: float
    max_fitness: float
    mean_fitness: float
    diversity: float          # 集団の平均ペアワイズ余弦距離
    centroid_dist: float      # 選択個体の重心の世代間コサイン距離（初世代は NaN）
    stagnant_streak: int      # 重心安定の連続世代数


@dataclass
class RunResult:
    approach: str
    param_name: str
    param_value: float
    target_prompt: str
    n_generations: int
    generation_logs: List[GenerationLog] = field(default_factory=list)
    # 収束速度: 最大適合度が 0.8 を超えた最初の世代（超えなければ n_generations）
    converge_generation: int = -1
    final_max_fitness: float = 0.0
    final_diversity: float = 0.0


# ---------------------------------------------------------------------------
# ベクトル演算ユーティリティ
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def pairwise_cosine_distance(embeddings: List[torch.Tensor]) -> float:
    """集団内の平均ペアワイズ余弦距離。"""
    vecs = [e.float().flatten() for e in embeddings]
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            dists.append(1.0 - sim)
    return float(np.mean(dists)) if dists else 0.0


def centroid_distance(prev: List[torch.Tensor], curr: List[torch.Tensor]) -> float:
    """2世代間の重心コサイン距離。"""
    c_prev = torch.stack([e.float().flatten() for e in prev]).mean(0)
    c_curr = torch.stack([e.float().flatten() for e in curr]).mean(0)
    return float(1.0 - F.cosine_similarity(c_prev.unsqueeze(0), c_curr.unsqueeze(0)).item())


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, threshold: float = 0.9995) -> torch.Tensor:
    """球面線形補間（任意形状テンソル対応）。"""
    shape = v0.shape
    a = v0.float().flatten()
    b = v1.float().flatten()
    a_n = a / (a.norm() + 1e-8)
    b_n = b / (b.norm() + 1e-8)
    dot = torch.clamp(torch.dot(a_n, b_n), -1.0, 1.0)
    if abs(dot.item()) > threshold:
        return ((1.0 - t) * a + t * b).reshape(shape).to(v0.dtype)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return (s0 * a + s1 * b).reshape(shape).to(v0.dtype)


# ---------------------------------------------------------------------------
# 代理適合度（CLAP embedding のコサイン類似度）
# ---------------------------------------------------------------------------

def compute_proxy_fitness(
    embeddings: List[torch.Tensor],
    target_embedding: torch.Tensor,
) -> List[float]:
    """各個体の embedding とターゲット embedding のコサイン類似度を返す。"""
    return [cosine_similarity(e, target_embedding) for e in embeddings]


# ---------------------------------------------------------------------------
# 選択（上位 k 体を選択）
# ---------------------------------------------------------------------------

def select_top_k(
    fitnesses: List[float],
    k: int = 2,
) -> List[int]:
    """適合度上位 k 体のインデックスを返す。"""
    indexed = sorted(enumerate(fitnesses), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]


# ---------------------------------------------------------------------------
# CSV/JSON 書き出し
# ---------------------------------------------------------------------------

def write_results_csv(results: List[RunResult], path: str) -> None:
    rows = []
    for r in results:
        for log in r.generation_logs:
            row = {
                "approach": r.approach,
                "param_name": r.param_name,
                "param_value": r.param_value,
                "target_prompt": r.target_prompt,
                **asdict(log),
            }
            rows.append(row)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp_common] CSV 出力: {path}")


def write_summary_json(results: List[RunResult], path: str) -> None:
    data = [asdict(r) for r in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[exp_common] JSON 出力: {path}")


# ---------------------------------------------------------------------------
# 実験ループ共通ヘルパー（モデルなしのシミュレーション用）
# ---------------------------------------------------------------------------

def simulate_gaussian_population(
    base_emb: torch.Tensor, sigma: float, n: int
) -> List[torch.Tensor]:
    """Gaussian approach: ベース embedding にガウシアンノイズを加算（非球面）。"""
    result = []
    for _ in range(n):
        noisy = base_emb + torch.randn_like(base_emb) * sigma
        result.append(noisy)
    return result


def simulate_slerp_random_population(
    base_emb: torch.Tensor, alpha: float, n: int
) -> List[torch.Tensor]:
    """SLERP B-1: 球面上の一様ランダムベクトルとの SLERP。"""
    result = []
    for _ in range(n):
        rand = torch.randn_like(base_emb)
        rand = rand / (rand.norm() + 1e-8)
        interp = slerp(base_emb, rand, alpha)
        result.append(interp)
    return result


def simulate_slerp_prompt_population(
    base_emb: torch.Tensor, pool_embs: List[torch.Tensor], alpha: float, n: int
) -> List[torch.Tensor]:
    """SLERP B-2: プール embedding との SLERP。"""
    result = []
    selected = random.sample(pool_embs, min(n, len(pool_embs)))
    while len(selected) < n:
        selected.extend(random.sample(pool_embs, min(n - len(selected), len(pool_embs))))
    for c_rand in selected[:n]:
        interp = slerp(base_emb, c_rand, alpha)
        result.append(interp)
    return result


def simulate_next_generation(
    population: List[torch.Tensor],
    selected_indices: List[int],
    pool_embs: List[torch.Tensor],
    elite_count: int = 2,
    random_sample_count: int = 1,
    p_mut: float = 0.4,
    mu_range: Tuple[float, float] = (0.05, 0.15),
    alpha_range: Tuple[float, float] = (0.3, 0.7),
    alpha_rand_range: Tuple[float, float] = (0.3, 0.6),
) -> List[torch.Tensor]:
    """設計仕様通りの次世代を SLERP ベースで生成する（モデル不要）。"""
    selected = [population[i] for i in selected_indices]
    pop_size = len(population)
    next_gen: List[torch.Tensor] = []

    # エリート
    actual_elite = min(elite_count, len(selected))
    next_gen.extend(selected[:actual_elite])

    # 交叉スロット
    actual_random = min(random_sample_count, pop_size - actual_elite)
    crossover_slots = pop_size - actual_elite - actual_random

    while len(next_gen) < actual_elite + crossover_slots:
        if len(selected) == 1:
            c_pool = random.choice(pool_embs)
            mu = float(np.random.uniform(*mu_range))
            child = slerp(selected[0], c_pool, mu)
        else:
            p1, p2 = random.sample(selected, 2)
            alpha = float(np.random.uniform(*alpha_range))
            child = slerp(p1, p2, alpha)
            if np.random.random() < p_mut:
                c_pool = random.choice(pool_embs)
                mu = float(np.random.uniform(*mu_range))
                child = slerp(child, c_pool, mu)
        next_gen.append(child)

    # ランダムサンプルスロット（SLERP B-2）
    base_emb = selected[0]
    for _ in range(actual_random):
        c_rand = random.choice(pool_embs)
        alpha_rand = float(np.random.uniform(*alpha_rand_range))
        next_gen.append(slerp(base_emb, c_rand, alpha_rand))

    return next_gen[:pop_size]


# ---------------------------------------------------------------------------
# プール embedding の事前生成（シミュレーション用）
# ---------------------------------------------------------------------------

def make_random_unit_embeddings(n: int, dim: int = 512) -> List[torch.Tensor]:
    """ランダムな単位ベクトルを n 件生成する（実モデルの代替）。"""
    result = []
    for _ in range(n):
        v = torch.randn(1, 1, dim)
        v = v / (v.norm() + 1e-8)
        result.append(v)
    return result
