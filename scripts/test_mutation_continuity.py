#!/usr/bin/env python3
"""
突然変異連続性テスト

検証内容:
  1. 連続性: mutation_strength を増やすと音声が段階的に変化するか
  2. 再現性: 同じ mutation_strength で異なるノイズシードを使っても傾向が安定するか

IEC mutation+SDEdit アプローチの有効性の前提条件を検証する。

Usage:
  NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \\
    python scripts/test_mutation_continuity.py [--prompt "text"] [--model audioldm-m-full]
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, ".")

from audioldm.iec_pipeline import AudioLDM_IEC

SAMPLE_RATE = 16000


def save_wav(path: str, wav: np.ndarray) -> None:
    wav = wav.squeeze()
    if wav.ndim > 1:
        wav = wav[0]
    sf.write(path, wav, SAMPLE_RATE)
    print(f"  saved: {os.path.basename(path)}  shape={wav.shape}")


def z0_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a.float() - b.float()).item())


def wav_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.squeeze().astype(np.float64)
    b = b.squeeze().astype(np.float64)
    if a.ndim > 1:
        a = a[0]
    if b.ndim > 1:
        b = b[0]
    min_len = min(len(a), len(b))
    return float(np.linalg.norm(a[:min_len] - b[:min_len]))


def mutate_z0(z0: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=z0.device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=g, device=z0.device, dtype=z0.dtype)
    return z0 + strength * noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="audioldm-m-full")
    parser.add_argument("--prompt", default="dog barking")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--sdedit_strength", type=float, default=0.1)
    parser.add_argument("--out_dir", default="output/mutation_continuity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cont_dir = os.path.join(args.out_dir, "continuity")
    repr_dir = os.path.join(args.out_dir, "reproducibility")
    os.makedirs(cont_dir, exist_ok=True)
    os.makedirs(repr_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"突然変異連続性テスト")
    print(f"  model          : {args.model}")
    print(f"  prompt         : {args.prompt}")
    print(f"  duration       : {args.duration}s")
    print(f"  sdedit_strength: {args.sdedit_strength}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"{'='*60}\n")

    pipe = AudioLDM_IEC(
        model_name=args.model,
        ga_mode="z0",
        duration=args.duration,
        ddim_steps=args.ddim_steps,
    )
    shape = (1,) + pipe.latent_shape

    # -------------------------------------------------------
    # ベース z0 を生成
    # -------------------------------------------------------
    print("[1/3] ベース z0 を生成中...")
    x_T_base = torch.randn(shape, device=pipe.device)
    z0_base = pipe._sample_z0_batch([x_T_base], text=args.prompt)[0]
    wav_base = pipe._decode_z0_batch([z0_base])[0]
    save_wav(os.path.join(args.out_dir, "base.wav"), wav_base)

    # -------------------------------------------------------
    # Test 1: 連続性テスト
    # -------------------------------------------------------
    print("\n[2/3] 連続性テスト: mutation_strength を変化させて SDEdit...")
    mutation_strengths = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    cont_results = []

    for strength in mutation_strengths:
        z0_mut = mutate_z0(z0_base, strength, seed=0)
        if strength == 0.0:
            z0_refined = z0_base
        else:
            z0_refined = pipe._sdedit_refine_z0_batch(
                [z0_mut], text=args.prompt, noise_strength=args.sdedit_strength
            )[0]
        wav_refined = pipe._decode_z0_batch([z0_refined])[0]

        fname = f"mut_{int(strength * 100):03d}_seed0.wav"
        save_wav(os.path.join(cont_dir, fname), wav_refined)

        l2_z0 = z0_l2(z0_base, z0_refined)
        l2_wav = wav_l2(wav_base, wav_refined)
        cont_results.append((strength, l2_z0, l2_wav))
        print(f"    strength={strength:.2f}  L2_z0={l2_z0:.4f}  L2_wav={l2_wav:.4f}")

    # -------------------------------------------------------
    # Test 2: 再現性テスト
    # -------------------------------------------------------
    print("\n[3/3] 再現性テスト: mutation_strength=0.2 で seed を変えて SDEdit...")
    repr_strength = 0.2
    repr_seeds = list(range(5))
    repr_l2_z0_list = []
    repr_l2_wav_list = []

    for seed in repr_seeds:
        z0_mut = mutate_z0(z0_base, repr_strength, seed=seed)
        z0_refined = pipe._sdedit_refine_z0_batch(
            [z0_mut], text=args.prompt, noise_strength=args.sdedit_strength
        )[0]
        wav_refined = pipe._decode_z0_batch([z0_refined])[0]

        fname = f"mut_{int(repr_strength * 100):03d}_seed{seed}.wav"
        save_wav(os.path.join(repr_dir, fname), wav_refined)

        l2_z0 = z0_l2(z0_base, z0_refined)
        l2_wav = wav_l2(wav_base, wav_refined)
        repr_l2_z0_list.append(l2_z0)
        repr_l2_wav_list.append(l2_wav)
        print(f"    seed={seed}  L2_z0={l2_z0:.4f}  L2_wav={l2_wav:.4f}")

    # -------------------------------------------------------
    # 結果サマリ
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print("Test 1: 連続性テスト結果")
    print(f"{'='*60}")
    print(f"  {'strength':>10}  {'L2_z0':>10}  {'L2_wav':>10}")
    for strength, l2_z0, l2_wav in cont_results:
        print(f"  {strength:>10.2f}  {l2_z0:>10.4f}  {l2_wav:>10.4f}")

    l2_z0_vals = [r[1] for r in cont_results]
    is_monotone = all(l2_z0_vals[i] <= l2_z0_vals[i + 1] for i in range(len(l2_z0_vals) - 1))
    print(f"\n  L2_z0 単調増加: {'YES → 連続性あり' if is_monotone else 'NO → 不連続の可能性あり'}")

    print(f"\n{'='*60}")
    print("Test 2: 再現性テスト結果 (mutation_strength=0.2)")
    print(f"{'='*60}")
    mean_z0 = float(np.mean(repr_l2_z0_list))
    std_z0 = float(np.std(repr_l2_z0_list))
    cv_z0 = std_z0 / mean_z0 if mean_z0 > 0 else float("inf")
    mean_wav = float(np.mean(repr_l2_wav_list))
    std_wav = float(np.std(repr_l2_wav_list))
    cv_wav = std_wav / mean_wav if mean_wav > 0 else float("inf")
    print(f"  L2_z0  mean={mean_z0:.4f}  std={std_z0:.4f}  CV={cv_z0:.3f}")
    print(f"  L2_wav mean={mean_wav:.4f}  std={std_wav:.4f}  CV={cv_wav:.3f}")
    print(f"\n  再現性: {'OK (CV < 0.3)' if cv_z0 < 0.3 else 'NG (CV >= 0.3)'}")

    print(f"\n{'='*60}")
    print("判断")
    print(f"{'='*60}")
    if is_monotone and cv_z0 < 0.3:
        print("  → IEC mutation+SDEdit は機能する可能性が高い")
    elif is_monotone:
        print("  → 連続性はあるが再現性が低い（同強度でも結果がバラつく）")
    elif cv_z0 < 0.3:
        print("  → 再現性はあるが連続性が低い（強度と変化量が対応しない）")
    else:
        print("  → 連続性・再現性ともに低い → ランダムサンプリングと同等の可能性")

    print(f"\n出力ディレクトリ: {args.out_dir}/")


if __name__ == "__main__":
    main()
