"""Exp 1: Conditioning Variation 手法の比較

3手法（Gaussian / SLERP B-1 / SLERP B-2）をターゲット固定で比較し、
収束速度・集団多様性・最終適合度を評価する。

使用方法:
    # ユニットテスト（モデル不要・シミュレーション）
    python scripts/exp1_conditioning_variation.py

    # 統合テスト（実モデル使用）
    python scripts/exp1_conditioning_variation.py --integration
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
    centroid_distance, slerp,
    simulate_gaussian_population, simulate_slerp_random_population,
    simulate_slerp_prompt_population, simulate_next_generation,
    make_random_unit_embeddings,
    write_results_csv, write_summary_json,
)

OUTPUT_DIR = "scripts/outputs/exp1"
N_GENERATIONS = 8
POP_SIZE = 6
SELECT_K = 2
N_TRIALS = 5
FITNESS_THRESHOLD = 0.8

APPROACHES = {
    "gaussian":     {"sigma": 0.1},
    "slerp_random": {"alpha": 0.3},
    "slerp_prompt": {"alpha": 0.3},
}


# ---------------------------------------------------------------------------
# シミュレーションモード（モデル不要）
# ---------------------------------------------------------------------------

def run_simulation(
    approach: str,
    param_value: float,
    target_emb: torch.Tensor,
    pool_embs: list,
    base_emb: torch.Tensor,
    n_gen: int = N_GENERATIONS,
    pop_size: int = POP_SIZE,
    trial: int = 0,
) -> RunResult:
    result = RunResult(
        approach=approach,
        param_name="alpha" if approach != "gaussian" else "sigma",
        param_value=param_value,
        target_prompt="[simulated]",
        n_generations=n_gen,
    )

    # 初期個体群を生成
    if approach == "gaussian":
        population = simulate_gaussian_population(base_emb, sigma=param_value, n=pop_size)
    elif approach == "slerp_random":
        population = simulate_slerp_random_population(base_emb, alpha=param_value, n=pop_size)
    else:  # slerp_prompt
        population = simulate_slerp_prompt_population(base_emb, pool_embs, alpha=param_value, n=pop_size)

    prev_selected_embs = None
    stagnant_streak = 0

    for gen in range(n_gen):
        fitnesses = compute_proxy_fitness(population, target_emb)
        selected_indices = select_top_k(fitnesses, k=SELECT_K)
        selected_embs = [population[i] for i in selected_indices]

        diversity = pairwise_cosine_distance(population)
        cdist = centroid_distance(prev_selected_embs, selected_embs) if prev_selected_embs else float("nan")
        if prev_selected_embs and cdist < 0.01:
            stagnant_streak += 1
        else:
            stagnant_streak = 0
        prev_selected_embs = selected_embs

        log = GenerationLog(
            generation=gen,
            approach=approach,
            param_name="alpha" if approach != "gaussian" else "sigma",
            param_value=param_value,
            max_fitness=max(fitnesses),
            mean_fitness=float(np.mean(fitnesses)),
            diversity=diversity,
            centroid_dist=cdist if not (isinstance(cdist, float) and np.isnan(cdist)) else -1.0,
            stagnant_streak=stagnant_streak,
        )
        result.generation_logs.append(log)

        if result.converge_generation < 0 and max(fitnesses) >= FITNESS_THRESHOLD:
            result.converge_generation = gen

        # 次世代を生成
        if gen < n_gen - 1:
            population = simulate_next_generation(
                population, selected_indices, pool_embs,
                elite_count=2, random_sample_count=1, p_mut=0.4,
            )

    last_fitnesses = compute_proxy_fitness(population, target_emb)
    result.final_max_fitness = max(last_fitnesses)
    result.final_diversity = pairwise_cosine_distance(population)
    if result.converge_generation < 0:
        result.converge_generation = n_gen  # 収束未達

    return result


def run_unit_tests() -> None:
    print("=" * 60)
    print("Exp 1: Conditioning Variation 比較（シミュレーション）")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    dim = 512
    target_emb = torch.randn(1, 1, dim)
    target_emb = target_emb / (target_emb.norm() + 1e-8)

    base_emb = torch.randn(1, 1, dim)
    base_emb = base_emb / (base_emb.norm() + 1e-8)

    pool_embs = make_random_unit_embeddings(50, dim)

    all_results: list[RunResult] = []

    for approach, params in APPROACHES.items():
        param_val = list(params.values())[0]
        trial_results = []
        for trial in range(N_TRIALS):
            r = run_simulation(approach, param_val, target_emb, pool_embs, base_emb, trial=trial)
            trial_results.append(r)
            all_results.append(r)

        avg_converge = np.mean([r.converge_generation for r in trial_results])
        avg_final_fit = np.mean([r.final_max_fitness for r in trial_results])
        avg_div = np.mean([r.final_diversity for r in trial_results])

        print(f"\n[{approach}] param={param_val}")
        print(f"  平均収束世代数 : {avg_converge:.1f}")
        print(f"  最終最大適合度 : {avg_final_fit:.4f}")
        print(f"  最終多様性     : {avg_div:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp1_simulation.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp1_simulation.json")
    print(f"\n結果を {OUTPUT_DIR}/ に保存しました。")

    # 合否判定: slerp_prompt が gaussian より低い収束世代数であること
    slerp_results = [r for r in all_results if r.approach == "slerp_prompt"]
    gauss_results = [r for r in all_results if r.approach == "gaussian"]
    sp_conv = np.mean([r.converge_generation for r in slerp_results])
    ga_conv = np.mean([r.converge_generation for r in gauss_results])
    print(f"\n仮説検証: slerp_prompt 収束世代 ({sp_conv:.1f}) <= gaussian ({ga_conv:.1f}): "
          f"{'PASS' if sp_conv <= ga_conv else 'FAIL (シミュレーション限界)'}")


# ---------------------------------------------------------------------------
# 統合テストモード（実モデル）
# ---------------------------------------------------------------------------

def run_integration(model_name: str, target_prompt: str, n_gen: int) -> None:
    print("=" * 60)
    print("Exp 1: Conditioning Variation 比較（統合テスト）")
    print(f"  モデル: {model_name}  ターゲット: {target_prompt}")
    print("=" * 60)

    from audioldm.iec_pipeline import AudioLDM_IEC
    from audioldm.prompt_pool import PROMPT_POOL, sample_prompts

    pipeline = AudioLDM_IEC(model_name=model_name, ga_mode="conditioning", population_size=POP_SIZE)

    target_emb = pipeline._encode_text_single(target_prompt)
    base_prompt = "music"
    base_emb = pipeline._encode_text_single(base_prompt)

    # プール CLAP embeddings を事前計算
    pool_prompts = sample_prompts(30, exclude=[base_prompt, target_prompt])
    pool_embs = [pipeline._encode_text_single(p).cpu() for p in pool_prompts]

    all_results: list[RunResult] = []

    for approach, params in APPROACHES.items():
        param_val = list(params.values())[0]
        print(f"\n--- {approach} (param={param_val}) ---")

        r = RunResult(
            approach=approach,
            param_name="alpha" if approach != "gaussian" else "sigma",
            param_value=param_val,
            target_prompt=target_prompt,
            n_generations=n_gen,
        )

        # 初期個体群
        results = pipeline.initialize_population_conditioning(
            prompt=base_prompt, slerp_alpha=param_val if approach == "slerp_prompt" else 0.0,
        )
        population_embs = [g.embedding.cpu() for g, _ in results]

        if approach == "gaussian":
            population_embs = simulate_gaussian_population(base_emb.cpu(), param_val, POP_SIZE)
        elif approach == "slerp_random":
            population_embs = simulate_slerp_random_population(base_emb.cpu(), param_val, POP_SIZE)

        prev_selected = None
        stagnant_streak = 0

        for gen in range(n_gen):
            fitnesses = compute_proxy_fitness(population_embs, target_emb.cpu())
            selected_indices = select_top_k(fitnesses, k=SELECT_K)
            selected_embs = [population_embs[i] for i in selected_indices]

            diversity = pairwise_cosine_distance(population_embs)
            cdist = centroid_distance(prev_selected, selected_embs) if prev_selected else -1.0
            if prev_selected and 0 <= cdist < 0.01:
                stagnant_streak += 1
            else:
                stagnant_streak = 0
            prev_selected = selected_embs

            log = GenerationLog(
                generation=gen, approach=approach,
                param_name=r.param_name, param_value=param_val,
                max_fitness=max(fitnesses), mean_fitness=float(np.mean(fitnesses)),
                diversity=diversity, centroid_dist=cdist,
                stagnant_streak=stagnant_streak,
            )
            r.generation_logs.append(log)
            print(f"  Gen {gen}: max_fitness={max(fitnesses):.4f}  diversity={diversity:.4f}")

            if r.converge_generation < 0 and max(fitnesses) >= FITNESS_THRESHOLD:
                r.converge_generation = gen

            if gen < n_gen - 1:
                population_embs = simulate_next_generation(
                    population_embs, selected_indices, pool_embs,
                    elite_count=2, random_sample_count=1, p_mut=0.4,
                )

        last_fitnesses = compute_proxy_fitness(population_embs, target_emb.cpu())
        r.final_max_fitness = max(last_fitnesses)
        r.final_diversity = pairwise_cosine_distance(population_embs)
        if r.converge_generation < 0:
            r.converge_generation = n_gen
        all_results.append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results_csv(all_results, f"{OUTPUT_DIR}/exp1_integration.csv")
    write_summary_json(all_results, f"{OUTPUT_DIR}/exp1_integration.json")
    print(f"\n結果を {OUTPUT_DIR}/ に保存しました。")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Conditioning Variation 比較")
    parser.add_argument("--integration", action="store_true", help="実モデルを使用する統合テストを実行")
    parser.add_argument("--model", default="audioldm-m-full", help="AudioLDM モデル名")
    parser.add_argument("--target", default="tense orchestral strings with brass fanfare",
                        help="ターゲット音声の説明プロンプト")
    parser.add_argument("--n-gen", type=int, default=N_GENERATIONS, help="世代数")
    args = parser.parse_args()

    if args.integration:
        run_integration(args.model, args.target, args.n_gen)
    else:
        run_unit_tests()
