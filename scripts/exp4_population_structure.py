"""Exp 4: 世代構成比率の比較

6体固定で (elite:crossover:random) = (2:3:1), (1:4:1), (2:2:2) を比較し、
収束速度と多様性への影響を評価する。

使用方法:
    python scripts/exp4_population_structure.py
    python scripts/exp4_population_structure.py --integration
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.exp_common import (
    GenerationLog, RunResult,
    compute_proxy_fitness, select_top_k, pairwise_cosine_distance,
    centroid_distance,
    simulate_slerp_prompt_population, simulate_next_generation,
    make_random_unit_embeddings,
    write_results_csv, write_summary_json,
)

OUTPUT_DIR = "scripts/outputs/exp4"

# (elite_count, crossover_slots, random_sample_count) の各構成
STRUCTURES: dict[str, Tuple[int, int, int]] = {
    "2:3:1": (2, 3, 1),
    "1:4:1": (1, 4, 1),
    "2:2:2": (2, 2, 2),
}
ALPHA = 0.3
P_MUT = 0.4
N_GENERATIONS = 8
POP_SIZE = 6
SELECT_K = 2
N_TRIALS = 5
FITNESS_THRESHOLD = 0.8


def run_simulation_structure(
    label: str,
    elite_count: int,
    random_sample_count: int,
    target_emb: torch.Tensor,
    pool_embs: list,
    base_emb: torch.Tensor,
    n_gen: int = N_GENERATIONS,
    pop_size: int = POP_SIZE,
) -> RunResult:
    result = RunResult(
        approach=label,
        param_name="structure",
        param_value=float(elite_count),
        target_prompt="[simulated]",
        n_generations=n_gen,
    )

    population = simulate_slerp_prompt_population(base_emb, pool_embs, alpha=ALPHA, n=pop_size)
    prev_selected = None
    stagnant_streak = 0

    for gen in range(n_gen):
        fitnesses = compute_proxy_fitness(population, target_emb)
        selected_indices = select_top_k(fitnesses, k=SELECT_K)
        selected_embs = [population[i] for i in selected_indices]

        diversity = pairwise_cosine_distance(population)
        cdist = centroid_distance(prev_selected, selected_embs) if prev_selected else -1.0
        if prev_selected and 0 <= cdist < 0.01:
            stagnant_streak += 1
        else:
            stagnant_streak = 0
        prev_selected = selected_embs

        log = GenerationLog(
            generation=gen, approach=label,
            param_name="structure", param_value=elite_count,
            max_fitness=max(fitnesses), mean_fitness=float(np.mean(fitnesses)),
            diversity=diversity, centroid_dist=cdist, stagnant_streak=stagnant_streak,
        )
        result.generation_logs.append(log)

        if result.converge_generation < 0 and max(fitnesses) >= FITNESS_THRESHOLD:
            result.converge_generation = gen

        if gen < n_gen - 1:
            population = simulate_next_generation(
                population, selected_indices, pool_embs,
                elite_count=elite_count,
                random_sample_count=random_sample_count,
                p_mut=P_MUT,
            )

    last_fitnesses = compute_proxy_fitness(population, target_emb)
    result.final_max_fitness = max(last_fitnesses)
    result.final_diversity = pairwise_cosine_distance(population)
    if result.converge_generation < 0:
        result.converge_generation = n_gen
    return result


def run_unit_tests() -> None:
    print("=" * 60)
    print("Exp 4: 世代構成比率の比較（シミュレーション）")
    print("=" * 60)

    torch.manual_seed(13)
    np.random.seed(13)

    dim = 512
    target_emb = torch.randn(1, 1, dim)
    target_emb = target_emb / (target_emb.norm() + 1e-8)
    base_emb = torch.randn(1, 1, dim)
    base_emb = base_emb / (base_emb.norm() + 1e-8)
    pool_embs = make_random_unit_embeddings(50, dim)

    all_results = []

    print(f"\n{'構成':>8} | {'収束世代':>8} | {'最終適合度':>10} | {'最終多様性':>10}")
    print("-" * 50)

    for label, (elite, crossover, random_sample) in STRUCTURES.items():
        trial_results = []
        for _ in range(N_TRIALS):
            r = run_simulation_structure(
                label, elite_count=elite, random_sample_count=random_sample,
                target_emb=target_emb, pool_embs=pool_embs, base_emb=base_emb,
            )
            trial_results.append(r)
            all_results.append(r)
        avg_c = np.mean([r.converge_generation for r in trial_results])
        avg_f = np.mean([r.final_max_fitness for r in trial_results])
        avg_d = np.mean([r.final_diversity for r in trial_results])
        print(f"{label:>8} | {avg_c:>8.1f} | {avg_f:>10.4f} | {avg_d:>10.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp4_simulation.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp4_simulation.json")

    # 推奨構成（設計仕様の標準 2:3:1）が他と比べてどうか
    std_results = [r for r in all_results if r.approach == "2:3:1"]
    best_conv = min(STRUCTURES.keys(), key=lambda l: np.mean(
        [r.converge_generation for r in all_results if r.approach == l]))
    print(f"\n最良収束構成: {best_conv}")
    print(f"結果を {OUTPUT_DIR}/ に保存しました。")


def run_integration(model_name: str, target_prompt: str, n_gen: int) -> None:
    print("=" * 60)
    print("Exp 4: 世代構成比率の比較（統合テスト）")
    print(f"  ターゲット: {target_prompt}")
    print("=" * 60)

    from audioldm.iec_pipeline import AudioLDM_IEC
    from audioldm.prompt_pool import sample_prompts

    pipeline = AudioLDM_IEC(model_name=model_name, ga_mode="conditioning", population_size=POP_SIZE)
    target_emb = pipeline._encode_text_single(target_prompt).cpu()
    base_emb = pipeline._encode_text_single("music").cpu()
    pool_prompts = sample_prompts(30, exclude=["music", target_prompt])
    pool_embs = [pipeline._encode_text_single(p).cpu() for p in pool_prompts]

    all_results = []
    for label, (elite, crossover, random_sample) in STRUCTURES.items():
        print(f"\n--- 構成 {label} ---")
        r = run_simulation_structure(
            label, elite_count=elite, random_sample_count=random_sample,
            target_emb=target_emb, pool_embs=pool_embs, base_emb=base_emb, n_gen=n_gen,
        )
        all_results.append(r)
        print(f"  収束世代: {r.converge_generation}  最終多様性: {r.final_diversity:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp4_integration.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp4_integration.json")
    print(f"\n結果を {OUTPUT_DIR}/ に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: 世代構成比率の比較")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--model", default="audioldm-m-full")
    parser.add_argument("--target", default="peaceful forest ambience with birds")
    parser.add_argument("--n-gen", type=int, default=N_GENERATIONS)
    args = parser.parse_args()

    if args.integration:
        run_integration(args.model, args.target, args.n_gen)
    else:
        run_unit_tests()
