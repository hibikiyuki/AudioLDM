"""音声長 × DDIM ステップ数を決めるための測定（2×4 の全組み合わせ）。

指導教員の試行で出た2つの指摘に答える。
  「1個体10秒は長い」「生成に20秒は長い」

**2つの梃子を直交させて測る。**

| | 生成長 | 提示長 |
|---|---|---|
| **LONG** | 10.24 秒（訓練長・品質が最良） | 先頭 5.12 秒を切り出し |
| **SHORT** | 5.12 秒ネイティブ（生成が速い） | そのまま 5.12 秒 |

× ステップ数 {200, 100, 50, 25}

**全条件を 5.12 秒・同一ノイズに揃える。**
SHORT 側は LONG 用ノイズ (T=256) の前半 (T=128) を使うので、
「短く生成したこと」と「ステップを削ったこと」だけが変数になる。

これで次の2つが同時に決まる。
  1. 短く生成して品質が落ちるか（LONG 対 SHORT）
  2. ステップを削って品質が落ちるか（200 → 100 → 50 → 25）
  3. **さらに「訓練長で生成しつつステップを削る」という組み合わせも評価できる**
     （B > A だった場合、これが唯一の時間短縮手段になる）

使用方法:
    NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
    python scripts/diag_musicldm_length_steps.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "ucsd-reach/musicldm"
OUT_DIR = Path("output/musicldm_length_steps")

PROMPTS = [
    ("p02", "fast punchy drum beat with groovy bass"),
    ("p03", "solo grand piano gentle melody"),
    ("p04", "aggressive electric guitar rock riff"),
    ("p10", "full orchestra with brass fanfare dramatic"),
]

LONG_S = 10.24     # 訓練長（潜在 T=256）
SHORT_S = 5.12     # 候補（潜在 T=128）。提示長もこれに揃える
STEP_SET = (200, 100, 50, 25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--bench-reps", type=int, default=3)
    args = ap.parse_args()

    import soundfile as sf
    from diffusers import MusicLDMPipeline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = MusicLDMPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    sr = pipe.vocoder.config.sampling_rate
    ch = pipe.unet.config.in_channels
    vuf = float(np.prod(pipe.vocoder.config.upsample_rates)) / sr

    def T_of(dur: float) -> int:
        return int(round((dur / vuf) / pipe.vae_scale_factor))

    T_long, T_short = T_of(LONG_S), T_of(SHORT_S)
    n_cut = int(SHORT_S * sr)
    print(f"LONG  {LONG_S}秒 → T={T_long}（生成後に先頭 {SHORT_S}秒を切り出し）")
    print(f"SHORT {SHORT_S}秒 → T={T_short}（LONG 用ノイズの前半 T=0:{T_short} を使う）")
    print(f"→ 全条件が {SHORT_S}秒・同一ノイズに揃う\n")

    def encode(text: str):
        tok = pipe.tokenizer([text], padding=True, truncation=True,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            return pipe.text_encoder.get_text_features(**tok)

    def gen(c, x_T, dur, steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe(prompt_embeds=c, latents=x_T, num_inference_steps=steps,
                       guidance_scale=args.guidance, audio_length_in_s=dur)
        return out.audios[0], time.perf_counter() - t0

    rows = []

    # ── 試聴サンプル（2 × 4 の全組み合わせ）──────────────────────────
    print("【試聴サンプルの生成】全条件 5.12 秒・同一ノイズ")
    for i, (pid, prompt) in enumerate(PROMPTS):
        c = encode(prompt)
        g = torch.Generator(device=device).manual_seed(3000 + i)
        x_full = torch.randn((1, ch, T_long, 16), generator=g, device=device)
        x_half = x_full[:, :, :T_short, :].clone()
        print(f"\n[{pid}] {prompt}")
        print(f"{'':6s} {'LONG(10.24秒生成→切出)':>26s} {'SHORT(5.12秒ネイティブ)':>24s}")
        for steps in STEP_SET:
            au, dt_l = gen(c, x_full, LONG_S, steps)
            sf.write(str(args.out_dir / f"{pid}_steps{steps:03d}_LONG.wav"), au[:n_cut], sr)
            if steps == STEP_SET[0]:   # 参考用に全長も残す
                sf.write(str(args.out_dir / f"{pid}_ref_{LONG_S:g}s_steps{steps}.wav"), au, sr)
            rows.append({"id": pid, "prompt": prompt, "cond": "LONG", "steps": steps,
                         "gen_s": LONG_S, "present_s": SHORT_S, "latent_T": T_long,
                         "gen_time_1": round(dt_l, 2)})

            au, dt_s = gen(c, x_half, SHORT_S, steps)
            sf.write(str(args.out_dir / f"{pid}_steps{steps:03d}_SHORT.wav"), au, sr)
            rows.append({"id": pid, "prompt": prompt, "cond": "SHORT", "steps": steps,
                         "gen_s": SHORT_S, "present_s": SHORT_S, "latent_T": T_short,
                         "gen_time_1": round(dt_s, 2)})
            print(f"  {steps:3d}  {dt_l:24.1f}秒 {dt_s:22.1f}秒")

    # ── 1世代6個体の生成時間（2 × 4 の全組み合わせ）───────────────
    print("\n【1世代6個体の生成時間（warmup 1回を捨てて中央値）】")
    print(f"{'生成長':>8s} {'steps':>6s} {'6個体':>8s} {'1個体':>7s}")
    bench = []
    c = encode(PROMPTS[0][1])
    for dur in (LONG_S, SHORT_S):
        T = T_of(dur)
        for steps in STEP_SET:
            times = []
            for k in range(args.bench_reps + 1):
                gg = torch.Generator(device=device).manual_seed(k)
                x = torch.randn((6, ch, T, 16), generator=gg, device=device)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    pipe(prompt_embeds=c.repeat(6, 1), latents=x,
                         num_inference_steps=steps, guidance_scale=args.guidance,
                         audio_length_in_s=dur)
                if device == "cuda":
                    torch.cuda.synchronize()
                if k:
                    times.append(time.perf_counter() - t0)
            m = statistics.median(times)
            print(f"{dur:7.2f}秒 {steps:6d} {m:7.1f}秒 {m/6:6.1f}秒")
            bench.append({"gen_s": dur, "steps": steps,
                          "batch6_median_s": round(m, 2),
                          "per_individual_s": round(m / 6, 2)})

    # ── 出力 ────────────────────────────────────────────────────────
    with open(args.out_dir / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(args.out_dir / "bench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bench[0].keys()))
        w.writeheader(); w.writerows(bench)

    bm = {(b["gen_s"], b["steps"]): b["batch6_median_s"] for b in bench}
    with open(args.out_dir / "manifest.md", "w", encoding="utf-8") as f:
        f.write("# 音声長 × DDIM ステップ数の決定\n\n")
        f.write("指摘「1個体10秒は長い」「生成に20秒は長い」への回答材料。\n\n")
        f.write("## 条件（2 × 4 の全組み合わせ）\n\n")
        f.write(f"**全条件が {SHORT_S:g} 秒・同一ノイズに揃っている。**\n")
        f.write(f"SHORT 側は LONG 用ノイズ (T={T_long}) の前半 (T={T_short}) を使うため、\n")
        f.write("「短く生成したこと」と「ステップを削ったこと」だけが変数になる。\n\n")
        f.write("| | 生成長 | 提示長 | 潜在 T |\n|---|---|---|---:|\n")
        f.write(f"| **LONG** | {LONG_S:g} 秒（訓練長） | 先頭 {SHORT_S:g} 秒を切り出し | {T_long} |\n")
        f.write(f"| **SHORT** | {SHORT_S:g} 秒ネイティブ | そのまま | {T_short} |\n\n")
        f.write("## 聴き方\n\n")
        f.write("**判断1：生成長** — 同じ steps で `LONG` と `SHORT` を比べる\n")
        f.write("- 差がなければ **SHORT を採れる**（生成が約半分）\n")
        f.write("- LONG が良ければ **訓練長で生成して提示だけ切る**\n\n")
        f.write("**判断2：ステップ数** — 同じ条件で 200 → 100 → 50 → 25 と聴く\n")
        f.write("- 劣化が分かる手前を採る。**生成時間はほぼステップ数に比例する**\n\n")
        f.write("**判断3：組み合わせ** — `LONG × 小ステップ` が使えるかを見る\n")
        f.write("- 判断1で LONG が良かった場合、**時間短縮の唯一の手段がステップ削減**になる\n")
        f.write("- 例: LONG×50 が LONG×200 と区別できなければ、品質を保ったまま 4 分の 1 に縮む\n\n")
        f.write("## 1世代6個体の生成時間\n\n")
        f.write("| 生成長 | steps=200 | 100 | 50 | 25 |\n|---:|---:|---:|---:|---:|\n")
        for dur in (LONG_S, SHORT_S):
            cells = " | ".join(f"{bm[(dur, s)]}秒" for s in STEP_SET)
            f.write(f"| {dur:.2f}秒 | {cells} |\n")
        f.write("\n## ファイル\n\n")
        for pid, prompt in PROMPTS:
            f.write(f"### {pid} `{prompt}`\n\n")
            f.write("| steps | LONG（10.24秒生成→切出） | SHORT（5.12秒ネイティブ） |\n|---:|---|---|\n")
            for steps in STEP_SET:
                f.write(f"| {steps} | `{pid}_steps{steps:03d}_LONG.wav` | "
                        f"`{pid}_steps{steps:03d}_SHORT.wav` |\n")
            f.write(f"\n参考（全長）: `{pid}_ref_{LONG_S:g}s_steps{STEP_SET[0]}.wav`\n\n")

    print(f"\n✅ 完了 / 試聴 {len(rows)} 件・計測 {len(bench)} 条件")
    print(f"   {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
