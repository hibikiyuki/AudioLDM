#!/usr/bin/env python3
"""
初期個体群多様性検証スクリプト

単一プロンプトからIEC初期個体群を生成する際の多様性注入アプローチを比較検証する。

アプローチ1 (Gaussian): c_i = c_base + N(0, σ²I)
アプローチ2 (SLERP):
  B-1: c_i = SLERP(c_base, c_rand_i, α)  c_rand_i = 乱数単位ベクトル（多様体外）
  B-2: c_i = SLERP(c_base, c_rand_i, α)  c_rand_i = ランダムプロンプトのCLAP埋め込み（多様体上）

x_T は全実験・全個体で完全固定し、conditioning vectorの変化のみを評価する。

実行方法:
  cd /workspaces/AudioLDM
  python scripts/test_initial_population_diversity.py

出力:
  scripts/outputs/initial_population_diversity/
    expA_gaussian_sigma{σ}_ind{i}.wav
    expB1_slerp_random_alpha{α}_ind{i}.wav
    expB2_slerp_prompt_alpha{α}_ind{i}.wav
    results.txt
"""

import argparse
import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# test_conditioning_genotype.py から importlib 経由で再利用
_tcg_path = os.path.join(os.path.dirname(__file__), "test_conditioning_genotype.py")
_spec = importlib.util.spec_from_file_location("test_conditioning_genotype", _tcg_path)
_tcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tcg)

ConditioningGenotype = _tcg.ConditioningGenotype
ConditioningGenotypeRunner = _tcg.ConditioningGenotypeRunner
slerp_conditioning = _tcg.slerp_conditioning
mutate_conditioning = _tcg.mutate_conditioning
save_wav = _tcg.save_wav
pairwise_cosine_mean = _tcg.pairwise_cosine_mean


# ──────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────

PROMPT_POOL = [
    "energetic electronic dance music",
    "heavy metal guitar riff",
    "ambient forest nature sounds",
    "jazz saxophone improvisation",
    "orchestral string quartet",
    "hip hop drum beat",
    "soft rain and thunder",
    "upbeat pop keyboard melody",
    "deep bass synthesizer",
    "acoustic folk guitar strumming",
    "children singing a lullaby",
    "fast tempo rock drum kit",
    "slow blues harmonica solo",
    "techno synthesizer arpeggio",
    "classical violin concerto",
]


# ──────────────────────────────────────────────────────────────
# 結果データクラス
# ──────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    approach: str           # "gaussian" | "slerp_random" | "slerp_prompt"
    param_name: str         # "sigma" | "alpha"
    param_value: float
    semantic_preservation: float    # c_i と c_base のcosine類似度の平均
    cond_diversity: float           # 条件付けベクトル間のペアワイズcosine距離の平均
    audio_diversity: float          # 音声CLAP埋め込み間のペアワイズcosine距離の平均
    degenerate_count: int           # 無音/NaN 個体の数
    n_individuals: int


# ──────────────────────────────────────────────────────────────
# 個体群ファクトリ
# ──────────────────────────────────────────────────────────────

class PopulationFactory:

    @staticmethod
    def gaussian_population(
        c_base: torch.Tensor,
        n: int,
        sigma: float,
        base_prompt: str = "",
    ) -> List[ConditioningGenotype]:
        """Approach 1: c_i = c_base + N(0, sigma^2 * I)"""
        g_base = ConditioningGenotype(embedding=c_base.clone(), source_prompt=base_prompt)
        genotypes = []
        for i in range(n):
            if sigma == 0.0:
                g = g_base.clone()
            else:
                g = mutate_conditioning(g_base, sigma=sigma)
            g.metadata.update({"approach": "gaussian", "sigma": sigma, "index": i})
            genotypes.append(g)
        return genotypes

    @staticmethod
    def slerp_random_population(
        c_base: torch.Tensor,
        n: int,
        alpha: float,
        device: str,
        base_prompt: str = "",
    ) -> List[ConditioningGenotype]:
        """Approach 2 B-1: c_i = SLERP(c_base, c_rand_i, alpha)
        c_rand_i: 乱数単位ベクトル（c_base と同ノルムに揃える）
        """
        base_norm = torch.norm(c_base.float()).item()
        genotypes = []
        for i in range(n):
            noise = torch.randn_like(c_base)
            c_rand = noise / (torch.norm(noise) + 1e-8) * base_norm
            c_rand = c_rand.to(c_base.dtype)
            c_i = slerp_conditioning(c_base, c_rand, alpha)
            g = ConditioningGenotype(
                embedding=c_i,
                source_prompt=f"random_unit_{i}",
                metadata={"approach": "slerp_random", "alpha": alpha, "index": i},
            )
            genotypes.append(g)
        return genotypes

    @staticmethod
    def slerp_prompt_population(
        c_base: torch.Tensor,
        n: int,
        alpha: float,
        runner: ConditioningGenotypeRunner,
        prompt_pool: List[str],
        base_prompt: str = "",
    ) -> List[ConditioningGenotype]:
        """Approach 2 B-2: c_i = SLERP(c_base, c_rand_prompt_i, alpha)
        c_rand_prompt_i: ランダムプロンプトをCLAPエンコードしたベクトル（多様体上）
        """
        sampled_prompts = random.choices(prompt_pool, k=n)
        genotypes = []
        for i, prompt in enumerate(sampled_prompts):
            c_rand = runner.encode_text(prompt)    # (1, 1, 512)
            c_i = slerp_conditioning(c_base, c_rand, alpha)
            g = ConditioningGenotype(
                embedding=c_i,
                source_prompt=prompt,
                metadata={"approach": "slerp_prompt", "alpha": alpha,
                          "rand_prompt": prompt, "index": i},
            )
            genotypes.append(g)
        return genotypes


# ──────────────────────────────────────────────────────────────
# 指標計算
# ──────────────────────────────────────────────────────────────

def is_degenerate(waveform: np.ndarray, silence_threshold: float = 1e-4) -> bool:
    if np.isnan(waveform).any():
        return True
    return float(np.max(np.abs(waveform))) < silence_threshold


def compute_metrics(
    c_base: torch.Tensor,
    genotypes: List[ConditioningGenotype],
    waveforms: List[np.ndarray],
    runner: ConditioningGenotypeRunner,
    approach: str,
    param_name: str,
    param_value: float,
) -> MetricResult:
    c_base_flat = c_base.reshape(1, -1).float()
    cond_embs = [g.embedding.reshape(1, -1).float() for g in genotypes]

    sims_to_base = [
        F.cosine_similarity(emb, c_base_flat, dim=1).item()
        for emb in cond_embs
    ]
    semantic_preservation = float(np.mean(sims_to_base))
    cond_diversity = pairwise_cosine_mean(cond_embs)

    degenerate_count = 0
    audio_embs = []
    for wf in waveforms:
        if is_degenerate(wf):
            degenerate_count += 1
        else:
            audio_embs.append(runner.encode_audio_embedding(wf))

    audio_diversity = pairwise_cosine_mean(audio_embs) if len(audio_embs) >= 2 else 0.0

    return MetricResult(
        approach=approach,
        param_name=param_name,
        param_value=param_value,
        semantic_preservation=semantic_preservation,
        cond_diversity=cond_diversity,
        audio_diversity=audio_diversity,
        degenerate_count=degenerate_count,
        n_individuals=len(genotypes),
    )


# ──────────────────────────────────────────────────────────────
# 実験ランナー
# ──────────────────────────────────────────────────────────────

def run_experiment_a(
    runner: ConditioningGenotypeRunner,
    x_T: torch.Tensor,
    c_base: torch.Tensor,
    base_prompt: str,
    out_dir: str,
    population_size: int = 6,
    sigmas: List[float] = (0.05, 0.1, 0.2, 0.5),
) -> List[MetricResult]:
    print("\n=== Experiment A: Gaussian Noise ===")
    results = []
    for sigma in sigmas:
        print(f"  sigma={sigma:.3f} で個体群を生成中...")
        t0 = time.time()
        genotypes = PopulationFactory.gaussian_population(
            c_base, n=population_size, sigma=sigma, base_prompt=base_prompt
        )
        cond_batch = torch.cat([g.embedding for g in genotypes], dim=0)  # (N, 1, 512)
        waveforms = runner.generate(cond_batch, x_T)
        elapsed = time.time() - t0

        for idx, wf in enumerate(waveforms):
            path = os.path.join(out_dir, f"expA_gaussian_sigma{sigma:.3f}_ind{idx:02d}.wav")
            save_wav(wf, path)

        metric = compute_metrics(
            c_base, genotypes, waveforms, runner,
            approach="gaussian", param_name="sigma", param_value=sigma
        )
        results.append(metric)
        print(
            f"    preservation={metric.semantic_preservation:.4f}  "
            f"cond_div={metric.cond_diversity:.4f}  "
            f"audio_div={metric.audio_diversity:.4f}  "
            f"degen={metric.degenerate_count}/{metric.n_individuals}  "
            f"({elapsed:.1f}s)"
        )
    return results


def run_experiment_b(
    runner: ConditioningGenotypeRunner,
    x_T: torch.Tensor,
    c_base: torch.Tensor,
    base_prompt: str,
    out_dir: str,
    population_size: int = 6,
    alphas: List[float] = (0.05, 0.1, 0.2, 0.3),
    prompt_pool: Optional[List[str]] = None,
) -> List[MetricResult]:
    print("\n=== Experiment B: SLERP ===")
    prompt_pool = prompt_pool or PROMPT_POOL
    results = []

    for alpha in alphas:
        # B-1: 乱数単位ベクトル（多様体外）
        print(f"  [B-1] alpha={alpha:.3f} (random unit vector) で生成中...")
        t0 = time.time()
        genotypes_b1 = PopulationFactory.slerp_random_population(
            c_base, n=population_size, alpha=alpha,
            device=runner.device, base_prompt=base_prompt
        )
        cond_batch_b1 = torch.cat([g.embedding for g in genotypes_b1], dim=0)
        waveforms_b1 = runner.generate(cond_batch_b1, x_T)
        elapsed = time.time() - t0

        for idx, wf in enumerate(waveforms_b1):
            path = os.path.join(out_dir, f"expB1_slerp_random_alpha{alpha:.3f}_ind{idx:02d}.wav")
            save_wav(wf, path)

        metric_b1 = compute_metrics(
            c_base, genotypes_b1, waveforms_b1, runner,
            approach="slerp_random", param_name="alpha", param_value=alpha
        )
        results.append(metric_b1)
        print(
            f"    preservation={metric_b1.semantic_preservation:.4f}  "
            f"cond_div={metric_b1.cond_diversity:.4f}  "
            f"audio_div={metric_b1.audio_diversity:.4f}  "
            f"degen={metric_b1.degenerate_count}/{metric_b1.n_individuals}  "
            f"({elapsed:.1f}s)"
        )

        # B-2: ランダムプロンプトのCLAP埋め込み（多様体上）
        print(f"  [B-2] alpha={alpha:.3f} (random prompt CLAP) で生成中...")
        t0 = time.time()
        genotypes_b2 = PopulationFactory.slerp_prompt_population(
            c_base, n=population_size, alpha=alpha,
            runner=runner, prompt_pool=prompt_pool, base_prompt=base_prompt
        )
        rand_prompts = [g.metadata.get("rand_prompt", "") for g in genotypes_b2]
        print(f"    ランダムプロンプト: {rand_prompts}")

        cond_batch_b2 = torch.cat([g.embedding for g in genotypes_b2], dim=0)
        waveforms_b2 = runner.generate(cond_batch_b2, x_T)
        elapsed = time.time() - t0

        for idx, wf in enumerate(waveforms_b2):
            path = os.path.join(out_dir, f"expB2_slerp_prompt_alpha{alpha:.3f}_ind{idx:02d}.wav")
            save_wav(wf, path)

        metric_b2 = compute_metrics(
            c_base, genotypes_b2, waveforms_b2, runner,
            approach="slerp_prompt", param_name="alpha", param_value=alpha
        )
        results.append(metric_b2)
        print(
            f"    preservation={metric_b2.semantic_preservation:.4f}  "
            f"cond_div={metric_b2.cond_diversity:.4f}  "
            f"audio_div={metric_b2.audio_diversity:.4f}  "
            f"degen={metric_b2.degenerate_count}/{metric_b2.n_individuals}  "
            f"({elapsed:.1f}s)"
        )

    return results


# ──────────────────────────────────────────────────────────────
# 結果出力
# ──────────────────────────────────────────────────────────────

_HDR = f"{'Approach':<14} {'Param':<6} {'Value':>6}  {'Sem.Pres':>8}  {'Cond.Div':>8}  {'Audio.Div':>9}  Degen"
_SEP = "-" * 70


def _row(r: MetricResult) -> str:
    return (
        f"{r.approach:<14} {r.param_name:<6} {r.param_value:>6.3f}  "
        f"{r.semantic_preservation:>8.4f}  "
        f"{r.cond_diversity:>8.4f}  "
        f"{r.audio_diversity:>9.4f}  "
        f"{r.degenerate_count}/{r.n_individuals}"
    )


def _recommendations(
    results: List[MetricResult],
    diversity_threshold: float,
    preservation_threshold: float,
) -> List[str]:
    lines = []
    by_approach = {}
    for r in results:
        by_approach.setdefault(r.approach, []).append(r)

    for approach, rows in sorted(by_approach.items()):
        good = [
            r for r in rows
            if r.cond_diversity >= diversity_threshold
            and r.semantic_preservation >= preservation_threshold
            and r.degenerate_count == 0
        ]
        if good:
            vals = [f"{r.param_value:.3f}" for r in good]
            lines.append(f"  {approach}: {rows[0].param_name} ∈ {{{', '.join(vals)}}}")
        else:
            lines.append(f"  {approach}: 条件を満たす値なし（閾値を下げるか探索範囲を変更してください）")
    return lines


def print_comparison_table(
    results: List[MetricResult],
    diversity_threshold: float = 0.05,
    preservation_threshold: float = 0.90,
):
    print("\n" + "=" * 70)
    print(" 初期個体群多様性比較 (diversity_thr={:.2f}, preservation_thr={:.2f})".format(
        diversity_threshold, preservation_threshold))
    print("=" * 70)
    print(_HDR)
    print(_SEP)
    prev_approach = None
    for r in results:
        if prev_approach and r.approach != prev_approach:
            print()
        print(_row(r))
        prev_approach = r.approach
    print(_SEP)
    print(f"推奨パラメータ範囲 (diversity>{diversity_threshold:.2f} AND preservation>{preservation_threshold:.2f} AND degen==0):")
    for line in _recommendations(results, diversity_threshold, preservation_threshold):
        print(line)
    print("=" * 70)


def save_results_txt(
    results: List[MetricResult],
    path: str,
    base_prompt: str,
    diversity_threshold: float,
    preservation_threshold: float,
    c_base_norm: float,
):
    lines = [
        "Initial Population Diversity Verification",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Prompt: \"{base_prompt}\"",
        f"Base embedding norm: {c_base_norm:.4f}",
        "",
        _HDR,
        _SEP,
    ]
    prev_approach = None
    for r in results:
        if prev_approach and r.approach != prev_approach:
            lines.append("")
        lines.append(_row(r))
        prev_approach = r.approach
    lines += [
        _SEP,
        "",
        f"推奨パラメータ範囲 (diversity>{diversity_threshold:.2f} AND preservation>{preservation_threshold:.2f} AND degen==0):",
    ]
    for line in _recommendations(results, diversity_threshold, preservation_threshold):
        lines.append(line)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n結果を保存しました: {path}")


# ──────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AudioLDM IEC 初期個体群多様性検証"
    )
    parser.add_argument(
        "--prompt", default="calm acoustic piano melody",
        help="ベースプロンプト"
    )
    parser.add_argument("--model_name", default="audioldm-m-full")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--ddim_steps", type=int, default=50,
        help="DDIMステップ数（検証用はデフォルト50）"
    )
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--population_size", type=int, default=6)
    parser.add_argument(
        "--noise_seed", type=int, default=42,
        help="全実験共通の固定 x_T のシード"
    )
    parser.add_argument(
        "--sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5],
        help="Experiment A: ガウスノイズの標準偏差リスト"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3],
        help="Experiment B: SLERP の補間係数リスト"
    )
    parser.add_argument("--diversity_threshold", type=float, default=0.05)
    parser.add_argument("--preservation_threshold", type=float, default=0.90)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs", "initial_population_diversity"
    )
    os.makedirs(out_dir, exist_ok=True)

    runner = ConditioningGenotypeRunner(
        model_name=args.model_name,
        ckpt_path=args.ckpt_path,
        duration=args.duration,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # 全実験・全個体で共有する固定 x_T
    x_T = runner.make_fixed_noise(seed=args.noise_seed)
    c_base = runner.encode_text(args.prompt)
    c_base_norm = torch.norm(c_base.float()).item()
    print(f"\nBase prompt: \"{args.prompt}\"")
    print(f"Base embedding norm: {c_base_norm:.4f}")
    print(f"Latent shape: {runner.latent_shape}")
    print(f"x_T shape: {x_T.shape}  (固定seed={args.noise_seed})")

    all_results: List[MetricResult] = []
    all_results += run_experiment_a(
        runner, x_T, c_base, args.prompt, out_dir,
        population_size=args.population_size,
        sigmas=args.sigmas,
    )
    all_results += run_experiment_b(
        runner, x_T, c_base, args.prompt, out_dir,
        population_size=args.population_size,
        alphas=args.alphas,
    )

    print_comparison_table(all_results, args.diversity_threshold, args.preservation_threshold)
    save_results_txt(
        all_results,
        os.path.join(out_dir, "results.txt"),
        base_prompt=args.prompt,
        diversity_threshold=args.diversity_threshold,
        preservation_threshold=args.preservation_threshold,
        c_base_norm=c_base_norm,
    )
    print(f"\n音声ファイル: {out_dir}/")


if __name__ == "__main__":
    main()
