"""Exp 3: Micro-SLERP 変異のアブレーション

p_mut ∈ {0.0, 0.3, 0.5} を変えて比較し、局所解停滞世代数への影響を測定する。

使用方法:
    python scripts/exp3_mutation_ablation.py
    python scripts/exp3_mutation_ablation.py --integration
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

OUTPUT_DIR = "scripts/outputs/exp3"
P_MUT_VALUES = [0.0, 0.3, 0.5]
ALPHA = 0.3
N_GENERATIONS = 8
POP_SIZE = 6
SELECT_K = 2
N_TRIALS = 5
FITNESS_THRESHOLD = 0.8
STAGNANT_EPS = 0.01  # ε: 重心安定判定閾値


def run_simulation_pmut(
    p_mut: float,
    target_emb: torch.Tensor,
    pool_embs: list,
    base_emb: torch.Tensor,
    n_gen: int = N_GENERATIONS,
    pop_size: int = POP_SIZE,
) -> RunResult:
    result = RunResult(
        approach=f"p_mut={p_mut}",
        param_name="p_mut",
        param_value=p_mut,
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
        if prev_selected and 0 <= cdist < STAGNANT_EPS:
            stagnant_streak += 1
        else:
            stagnant_streak = 0
        prev_selected = selected_embs

        log = GenerationLog(
            generation=gen, approach=result.approach,
            param_name="p_mut", param_value=p_mut,
            max_fitness=max(fitnesses), mean_fitness=float(np.mean(fitnesses)),
            diversity=diversity, centroid_dist=cdist, stagnant_streak=stagnant_streak,
        )
        result.generation_logs.append(log)

        if result.converge_generation < 0 and max(fitnesses) >= FITNESS_THRESHOLD:
            result.converge_generation = gen

        if gen < n_gen - 1:
            population = simulate_next_generation(
                population, selected_indices, pool_embs,
                elite_count=2, random_sample_count=1, p_mut=p_mut,
            )

    last_fitnesses = compute_proxy_fitness(population, target_emb)
    result.final_max_fitness = max(last_fitnesses)
    result.final_diversity = pairwise_cosine_distance(population)
    if result.converge_generation < 0:
        result.converge_generation = n_gen
    return result


def run_unit_tests() -> None:
    print("=" * 60)
    print("Exp 3: Micro-SLERP 変異アブレーション（シミュレーション）")
    print("=" * 60)

    torch.manual_seed(7)
    np.random.seed(7)

    dim = 512
    target_emb = torch.randn(1, 1, dim)
    target_emb = target_emb / (target_emb.norm() + 1e-8)
    base_emb = torch.randn(1, 1, dim)
    base_emb = base_emb / (base_emb.norm() + 1e-8)
    pool_embs = make_random_unit_embeddings(50, dim)

    all_results = []

    print(f"\n{'p_mut':>8} | {'収束世代':>8} | {'最終適合度':>10} | {'最終多様性':>10} | {'最長停滞':>8}")
    print("-" * 60)

    for p_mut in P_MUT_VALUES:
        trial_results = []
        for _ in range(N_TRIALS):
            r = run_simulation_pmut(p_mut, target_emb, pool_embs, base_emb)
            trial_results.append(r)
            all_results.append(r)

        avg_c = np.mean([r.converge_generation for r in trial_results])
        avg_f = np.mean([r.final_max_fitness for r in trial_results])
        avg_d = np.mean([r.final_diversity for r in trial_results])
        # 各試行で最大の stagnant_streak を取得
        max_stagnant = float(np.mean([
            max((log.stagnant_streak for log in r.generation_logs), default=0)
            for r in trial_results
        ]))
        print(f"{p_mut:>8.1f} | {avg_c:>8.1f} | {avg_f:>10.4f} | {avg_d:>10.4f} | {max_stagnant:>8.1f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp3_simulation.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp3_simulation.json")
    print(f"\n結果を {OUTPUT_DIR}/ に保存しました。")

    # 仮説: p_mut > 0 の場合、最長停滞が p_mut=0 より短い
    no_mut = [r for r in all_results if r.param_value == 0.0]
    with_mut = [r for r in all_results if r.param_value > 0.0]
    no_mut_stag = np.mean([max((l.stagnant_streak for l in r.generation_logs), default=0) for r in no_mut])
    with_mut_stag = np.mean([max((l.stagnant_streak for l in r.generation_logs), default=0) for r in with_mut])
    print(f"\n仮説検証: 変異あり停滞 ({with_mut_stag:.2f}) <= 変異なし ({no_mut_stag:.2f}): "
          f"{'PASS' if with_mut_stag <= no_mut_stag else 'FAIL (シミュレーション限界)'}")


def run_integration(model_name: str, target_prompt: str, n_gen: int) -> None:
    print("=" * 60)
    print("Exp 3: Micro-SLERP 変異アブレーション（統合テスト）")
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
    for p_mut in P_MUT_VALUES:
        print(f"\n--- p_mut={p_mut} ---")
        r = run_simulation_pmut(p_mut, target_emb, pool_embs, base_emb, n_gen=n_gen)
        all_results.append(r)
        max_stag = max((l.stagnant_streak for l in r.generation_logs), default=0)
        print(f"  収束世代: {r.converge_generation}  最長停滞: {max_stag}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp3_integration.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp3_integration.json")
    print(f"\n結果を {OUTPUT_DIR}/ に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Micro-SLERP 変異アブレーション")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--model", default="audioldm-m-full")
    parser.add_argument("--target", default="energetic electronic dance music with heavy bass")
    parser.add_argument("--n-gen", type=int, default=N_GENERATIONS)
    args = parser.parse_args()

    if args.integration:
        run_integration(args.model, args.target, args.n_gen)
    else:
        run_unit_tests()
