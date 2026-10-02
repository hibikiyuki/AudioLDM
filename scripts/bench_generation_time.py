#!/usr/bin/env python
"""生成時間のベンチマーク。

発表・論文で「1世代6個体の生成に約◯秒、GPU は◯◯」と数字で言えるようにするための計測。
展示・ユーザスタディと同じ設定（audioldm-m-full / 2.5秒 / DDIM 200 / guidance 2.5 / 6個体）が既定。

    python scripts/bench_generation_time.py

注意: 1世代の6個体は `_sample_audio_batch` でバッチ生成され、DDIM は1回しか回らない。
      そのため「1個体あたり」は 6個体の所要時間を割った値（償却値）であり、
      1個体だけ生成したときの時間とは一致しない。両方を計測して併記する。
"""

import argparse
import statistics
import time

import torch

from audioldm.iec_pipeline import AudioLDM_IEC


def bench(iec: AudioLDM_IEC, batch_size: int, repeats: int, warmup: int, prompt: str):
    """指定バッチサイズで生成し、所要秒数のリストを返す。"""
    times = []
    for i in range(warmup + repeats):
        x_T_list = [
            torch.randn(1, *iec.latent_shape, device=iec.device)
            for _ in range(batch_size)
        ]
        if iec.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iec._sample_audio_batch(x_T_list, text=prompt)
        if iec.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tag = "warmup" if i < warmup else f"{i - warmup + 1}/{repeats}"
        print(f"  [{batch_size}個体] {tag}: {elapsed:.2f} 秒")
        if i >= warmup:
            times.append(elapsed)
    return times


def summarize(times):
    return {
        "median": statistics.median(times),
        "mean": statistics.fmean(times),
        "min": min(times),
        "max": max(times),
    }


def main():
    p = argparse.ArgumentParser(description="AudioLDM 生成時間のベンチマーク")
    p.add_argument("--model_name", type=str, default="audioldm-m-full")
    p.add_argument("--duration", type=float, default=2.5)
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--population_size", type=int, default=6)
    p.add_argument("--repeats", type=int, default=5, help="計測回数（warmup を除く）")
    p.add_argument("--warmup", type=int, default=1, help="捨てる試行数。初回は CUDA 初期化を含む")
    p.add_argument("--prompt", type=str, default="electric guitar hard rock")
    args = p.parse_args()

    print("=" * 62)
    print(" 生成時間ベンチマーク")
    print("=" * 62)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU        : {gpu_name}（VRAM {vram_gb:.0f} GB）")
    else:
        gpu_name, vram_gb = None, None
        print("GPU        : なし（CPU 実行。発表用の数字には使えない）")
    print(f"torch      : {torch.__version__}")
    print(f"モデル     : {args.model_name}")
    print(f"音声長     : {args.duration} 秒")
    print(f"DDIM steps : {args.ddim_steps}")
    print(f"guidance   : {args.guidance_scale}")
    print(f"個体数     : {args.population_size}")
    print("-" * 62)

    iec = AudioLDM_IEC(
        model_name=args.model_name,
        population_size=args.population_size,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
    )

    print("\n[1] 1世代ぶん（バッチ生成）")
    gen_times = bench(iec, args.population_size, args.repeats, args.warmup, args.prompt)
    gen = summarize(gen_times)

    print("\n[2] 1個体のみ（バッチサイズ1）")
    one_times = bench(iec, 1, args.repeats, args.warmup, args.prompt)
    one = summarize(one_times)

    per_ind = gen["median"] / args.population_size

    print("\n" + "=" * 62)
    print(" 結果（中央値）")
    print("=" * 62)
    print(f"1世代 {args.population_size} 個体 : {gen['median']:.1f} 秒"
          f"（平均 {gen['mean']:.1f} / 最小 {gen['min']:.1f} / 最大 {gen['max']:.1f}）")
    print(f"1個体あたり（償却）: {per_ind:.1f} 秒")
    print(f"1個体のみ生成      : {one['median']:.1f} 秒"
          f"（平均 {one['mean']:.1f} / 最小 {one['min']:.1f} / 最大 {one['max']:.1f}）")
    print(f"バッチ化による短縮 : {one['median'] * args.population_size / gen['median']:.1f} 倍")

    print("\n--- 発表用の一文 ---")
    gpu_label = gpu_name if gpu_name else "（GPU 未検出）"
    print(
        f"音声長 {args.duration} 秒、DDIM {args.ddim_steps} ステップで、"
        f"1世代 {args.population_size} 個体の生成に約 {gen['median']:.0f} 秒です。"
        f"1個体あたりに直すと約 {per_ind:.1f} 秒。GPU は {gpu_label} です。"
    )
    print("-" * 62)
    print("※ 8世代ぶんの生成待ち時間の合計は "
          f"約 {gen['median'] * 8 / 60:.1f} 分（{gen['median'] * 8:.0f} 秒）")


if __name__ == "__main__":
    main()
