"""MusicLDM の短尺で品質が落ちる原因を切り分ける。

「2.5 秒だと音色の質が落ちる、10 秒の品質が良い」という観察に対し、
原因候補を2つに分けて検証する。

**原因候補1：潜在格子との不整合**
  UNet の時間軸ダウンサンプル幅は 8。潜在の時間長 T が 8 の倍数でないと
  各解像度でパディング／クロップが生じる。
    2.50 秒 → T=62（8 の倍数でない。余り 6）  ← 前回生成したのはこれ
    2.56 秒 → T=64（8 の倍数）
  なお AudioLDM は `latent_t_size = int(duration * 25.6)` なので
  2.5 秒 → T=64 で整合していた。「2.5 秒の倍数のみ有効」という制約は
  まさに T を 8 の倍数に乗せるためのものだった。
  **MusicLDM では 25 フレーム/秒なので、2.5 秒は整合しない。**

**原因候補2：訓練分布からの逸脱**
  `unet.config.sample_size = 256` → 訓練時の潜在時間長は T=256（= 10.24 秒）。
  T=64 は訓練長の 1/4 で、分布から外れている。

切り分け方：
  A 2.50 秒ネイティブ（T=62・不整合）
  B 2.56 秒ネイティブ（T=64・整合）
  C 10.24 秒を生成して先頭 2.56 秒を切り出す（整合かつ訓練長で生成）
  D 10.24 秒そのまま（訓練長）

  B > A なら原因は**不整合**。C > B なら原因は**訓練分布からの逸脱**。
  C ≈ D なら「長く生成して切り出す」が実用的な解になる。

使用方法:
    NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
    python scripts/diag_musicldm_duration.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "ucsd-reach/musicldm"
OUT_DIR = Path("output/musicldm_duration_diag")

# 品質差が聴き取りやすいものを選ぶ
PROMPTS = [
    ("p02", "fast punchy drum beat with groovy bass"),
    ("p03", "solo grand piano gentle melody"),
    ("p04", "aggressive electric guitar rock riff"),
    ("p10", "full orchestra with brass fanfare dramatic"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import soundfile as sf
    from diffusers import MusicLDMPipeline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = MusicLDMPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device).to(device)
    pipe.set_progress_bar_config(disable=True)

    sr = pipe.vocoder.config.sampling_rate
    ch = pipe.unet.config.in_channels
    vuf = float(np.prod(pipe.vocoder.config.upsample_rates)) / sr
    nds = 2 ** (len(pipe.unet.config.block_out_channels) - 1)
    train_T = pipe.unet.config.sample_size

    def T_of(dur: float) -> int:
        return int(round((dur / vuf) / pipe.vae_scale_factor))

    print(f"訓練時の潜在時間長 T={train_T}（= {train_T * pipe.vae_scale_factor * vuf:.2f} 秒）")
    print(f"UNet 時間軸ダウンサンプル幅 = {nds}\n")

    def encode(text: str):
        tok = pipe.tokenizer([text], padding=True, truncation=True,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            return pipe.text_encoder.get_text_features(**tok)

    def gen(c, dur: float, seed: int):
        T = T_of(dur)
        g = torch.Generator(device=device).manual_seed(seed)
        x_T = torch.randn((1, ch, T, 16), generator=g, device=device)
        t0 = time.time()
        with torch.no_grad():
            out = pipe(prompt_embeds=c, latents=x_T,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance, audio_length_in_s=dur)
        return out.audios[0], T, time.time() - t0

    rows = []
    for i, (pid, prompt) in enumerate(PROMPTS):
        c = encode(prompt)
        seed = 2000 + i
        print(f"[{pid}] {prompt}")

        # A: 2.50 秒ネイティブ（不整合 T=62）
        a, Ta, ta = gen(c, 2.50, seed)
        sf.write(str(args.out_dir / f"{pid}_A_2.50s_T{Ta}_native.wav"), a, sr)
        print(f"   A 2.50秒 native  T={Ta:3d} ({'整合' if Ta % nds == 0 else '不整合'})  {ta:.1f}秒")

        # B: 2.56 秒ネイティブ（整合 T=64）
        b, Tb, tb = gen(c, 2.56, seed)
        sf.write(str(args.out_dir / f"{pid}_B_2.56s_T{Tb}_native.wav"), b, sr)
        print(f"   B 2.56秒 native  T={Tb:3d} ({'整合' if Tb % nds == 0 else '不整合'})  {tb:.1f}秒")

        # D: 10.24 秒ネイティブ（訓練長）
        d, Td, td = gen(c, 10.24, seed)
        sf.write(str(args.out_dir / f"{pid}_D_10.24s_T{Td}_native.wav"), d, sr)
        print(f"   D 10.24秒 native T={Td:3d} ({'整合' if Td % nds == 0 else '不整合'})  {td:.1f}秒")

        # C: D の先頭 2.56 秒を切り出す
        n = int(2.56 * sr)
        cpart = d[:n]
        sf.write(str(args.out_dir / f"{pid}_C_2.56s_trunc_from_10.24.wav"), cpart, sr)
        print(f"   C 10.24秒を生成して先頭 2.56 秒を切り出し（追加の生成コストなし）")

        for tag, dur, T, arr, note in (
            ("A", 2.50, Ta, a, "ネイティブ・潜在格子と不整合"),
            ("B", 2.56, Tb, b, "ネイティブ・潜在格子と整合"),
            ("C", 2.56, Td, cpart, "10.24 秒を生成して切り出し"),
            ("D", 10.24, Td, d, "訓練長そのまま"),
        ):
            rows.append({"id": pid, "cond": tag, "prompt": prompt,
                         "target_s": dur, "latent_T": T,
                         "aligned": T % nds == 0, "actual_s": round(len(arr) / sr, 2),
                         "note": note})

    with open(args.out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(args.out_dir / "manifest.md", "w", encoding="utf-8") as f:
        f.write("# MusicLDM 音声長の切り分け\n\n")
        f.write(f"- 訓練時の潜在時間長 **T={train_T}（= 10.24 秒）**"
                f"（`unet.config.sample_size`）\n")
        f.write(f"- UNet の時間軸ダウンサンプル幅 **{nds}** → T がその倍数でないと"
                f"各解像度でパディング／クロップが生じる\n")
        f.write(f"- MusicLDM の潜在は **25 フレーム/秒** なので "
                f"**2.5 秒は T=62 で整合しない**。2.56 秒なら T=64 で整合する\n")
        f.write(f"- AudioLDM は 25.6 フレーム/秒なので 2.5 秒 → T=64 で整合していた"
                f"（「2.5 秒の倍数のみ」という制約の正体）\n\n")
        f.write("## 比較する4条件（同一プロンプト・同一 seed）\n\n")
        f.write("| 条件 | 内容 | 潜在 T | 整合 |\n|---|---|---:|---|\n")
        f.write(f"| **A** | 2.50 秒ネイティブ（前回生成したもの） | 62 | ❌ |\n")
        f.write(f"| **B** | 2.56 秒ネイティブ | 64 | ✅ |\n")
        f.write(f"| **C** | 10.24 秒を生成して先頭 2.56 秒を切り出し | 256 | ✅ |\n")
        f.write(f"| **D** | 10.24 秒そのまま（訓練長） | 256 | ✅ |\n\n")
        f.write("## 読み方\n\n")
        f.write("- **B が A より良い** → 原因は潜在格子との**不整合**。"
                "音声長を 2.56 秒に変えるだけで解決する\n")
        f.write("- **C が B より良い** → 原因は**訓練分布からの逸脱**。"
                "短く生成すること自体が品質を落としている\n")
        f.write("- **C ≈ D** → 「長く生成して切り出す」が実用的な解。"
                "ただし生成コストは 10.24 秒ぶんかかる\n\n")
        f.write("## ファイル\n\n")
        for pid, prompt in PROMPTS:
            f.write(f"### {pid} `{prompt}`\n\n")
            f.write(f"- A `{pid}_A_2.50s_T62_native.wav`\n")
            f.write(f"- B `{pid}_B_2.56s_T64_native.wav`\n")
            f.write(f"- C `{pid}_C_2.56s_trunc_from_10.24.wav`\n")
            f.write(f"- D `{pid}_D_10.24s_T256_native.wav`\n\n")

    print(f"\n✅ 完了 / {len(rows)} 件")
    print(f"   {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
