#!/usr/bin/env python3
"""
SDEdit音質テスト

検証内容:
  1. サニティチェック: クリーンなz0にSDEditを適用しても大きく変化しない
  2. 交叉テスト: SLERP交叉で得られた子z0にSDEditを適用すると音質が改善される

Usage:
  NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
    python scripts/test_sdedit_quality.py [--prompt "text"] [--model audioldm-m-full]
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, ".")

from audioldm.iec import LatentZ0Genotype, crossover_z0_slerp
from audioldm.iec_pipeline import AudioLDM_IEC

SAMPLE_RATE = 16000


def save_wav(path: str, wav: np.ndarray) -> float:
    wav = wav.squeeze()
    if wav.ndim > 1:
        wav = wav[0]
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    sf.write(path, wav, SAMPLE_RATE)
    print(f"  saved: {os.path.basename(path)}  rms={rms:.4f}  shape={wav.shape}")
    return rms


def z0_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a.float() - b.float()).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="audioldm-m-full")
    parser.add_argument("--prompt_a", default="dog barking")
    parser.add_argument("--prompt_b", default="piano music")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--out_dir", default="output/sdedit_test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"SDEdit音質テスト")
    print(f"  model   : {args.model}")
    print(f"  prompt_a: {args.prompt_a}")
    print(f"  prompt_b: {args.prompt_b}")
    print(f"  duration: {args.duration}s")
    print(f"  out_dir : {args.out_dir}")
    print(f"{'='*60}\n")

    pipe = AudioLDM_IEC(
        model_name=args.model,
        ga_mode="z0",
        duration=args.duration,
        ddim_steps=args.ddim_steps,
    )
    shape = (1,) + pipe.latent_shape

    # -------------------------------------------------------
    # Step 1: 親2体のz0を生成
    # -------------------------------------------------------
    print("[1/4] 親個体A・Bのz0を生成中...")
    x_T_a = torch.randn(shape, device=pipe.device)
    x_T_b = torch.randn(shape, device=pipe.device)
    z0_a = pipe._sample_z0_batch([x_T_a], text=args.prompt_a)[0]
    z0_b = pipe._sample_z0_batch([x_T_b], text=args.prompt_b)[0]

    wav_a = pipe._decode_z0_batch([z0_a])[0]
    wav_b = pipe._decode_z0_batch([z0_b])[0]
    save_wav(f"{args.out_dir}/parent_a.wav", wav_a)
    save_wav(f"{args.out_dir}/parent_b.wav", wav_b)

    # -------------------------------------------------------
    # Step 2: SLERP交叉で子z0を生成
    # -------------------------------------------------------
    print("\n[2/4] SLERP交叉で子z0を生成中 (alpha=0.5)...")
    g_a = LatentZ0Genotype(z0=z0_a.cpu())
    g_b = LatentZ0Genotype(z0=z0_b.cpu())
    child = crossover_z0_slerp(g_a, g_b, alpha=0.5)
    z0_child = child.z0.to(pipe.device)

    wav_child_raw = pipe._decode_z0_batch([z0_child])[0]
    save_wav(f"{args.out_dir}/child_raw.wav", wav_child_raw)

    # -------------------------------------------------------
    # Step 3: 子z0にSDEditを各強度で適用
    # -------------------------------------------------------
    print("\n[3/4] 子z0にSDEditを適用中...")
    sdedit_strengths = [0.1, 0.2, 0.4, 0.6, 0.8]
    z0_child_refined = {}
    for s in sdedit_strengths:
        print(f"  noise_strength={s}")
        refined = pipe._sdedit_refine_z0_batch(
            [z0_child], text=args.prompt_a, noise_strength=s
        )[0]
        z0_child_refined[s] = refined
        wav = pipe._decode_z0_batch([refined])[0]
        save_wav(f"{args.out_dir}/child_sdedit_{int(s*10):02d}.wav", wav)

    # -------------------------------------------------------
    # Step 4: サニティチェック（クリーンなz0にSDEdit）
    # -------------------------------------------------------
    print("\n[4/4] サニティチェック: クリーンなz0にSDEdit strength=0.1を適用...")
    z0_sanity = pipe._sdedit_refine_z0_batch(
        [z0_a], text=args.prompt_a, noise_strength=0.1
    )[0]
    wav_sanity = pipe._decode_z0_batch([z0_sanity])[0]
    save_wav(f"{args.out_dir}/sanity_sdedit.wav", wav_sanity)

    # -------------------------------------------------------
    # 統計表示
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print("z0空間でのL2距離比較")
    print(f"{'='*60}")
    print(f"  L2(parent_a, parent_b)       = {z0_l2(z0_a, z0_b):.4f}  (2つの独立サンプル)")
    print(f"  L2(parent_a, sanity_sdedit)  = {z0_l2(z0_a, z0_sanity):.4f}  (サニティ: 小さければOK)")
    print(f"  L2(parent_a, child_raw)      = {z0_l2(z0_a, z0_child):.4f}  (交叉子との距離)")
    for s, z0r in z0_child_refined.items():
        print(f"  L2(child_raw, sdedit_{s})    = {z0_l2(z0_child, z0r):.4f}  (SDEditによる変化量)")

    print(f"\n{'='*60}")
    print("判断基準")
    print(f"{'='*60}")
    sanity_l2 = z0_l2(z0_a, z0_sanity)
    ab_l2 = z0_l2(z0_a, z0_b)
    print(f"  [{'OK' if sanity_l2 < ab_l2 * 0.2 else 'NG'}] サニティL2 < 親間距離×0.2: {sanity_l2:.4f} < {ab_l2*0.2:.4f}")
    print(f"  →  聴感: parent_a.wav ≈  sanity_sdedit.wav なら SDEditは元音を保持している")
    print(f"  →  聴感: child_raw.wav より child_sdedit_02.wav の方が自然なら SDEdit有効")
    print(f"\n出力ディレクトリ: {args.out_dir}/")


if __name__ == "__main__":
    main()
