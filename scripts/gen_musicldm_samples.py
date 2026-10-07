"""MusicLDM の試聴サンプルを生成する（音声長の判断材料）。

目的：**MusicLDM の 2.5 秒が音楽として体裁を保つか**を耳で判断するための材料を作る。
AudioLDM（一般音響モデル）では 2.5 秒が「音楽といえない」と指摘されたが、
音楽特化モデルなら成立する可能性がある。それを確かめる。

同じプロンプト・同じ seed で 2.5 秒と 10 秒を生成して並べるので、
短くすることで何が失われるかを直接比較できる。

プロンプトは次を混ぜてある：
  - テンポと音の密度の極端（遅い持続音／速いリズム／単一楽器の旋律）
  - ジャンル（ロック・ホラー・ジャズ・バロック・EDM・lofi・オーケストラ）
  - パイロットのお題に対応する方向（格闘ゲーム＝激しい／ホラー地下室＝暗い）

使用方法:
    NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
    python scripts/gen_musicldm_samples.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "ucsd-reach/musicldm"
OUT_DIR = Path("output/musicldm_samples")

# (ID, プロンプト, 狙い)
PROMPTS = [
    ("p01", "slow ambient pad with sustained strings",
     "遅い・持続音。短尺で最も有利な条件"),
    ("p02", "fast punchy drum beat with groovy bass",
     "速い・リズム密。2.5秒にフレーズが入るか"),
    ("p03", "solo grand piano gentle melody",
     "単一楽器の旋律。2.5秒で旋律が成立するか"),
    ("p04", "aggressive electric guitar rock riff",
     "激しい系。お題「対戦格闘ゲーム」に対応"),
    ("p05", "dark eerie horror ambience with low drone",
     "暗い系。お題「ホラーゲームの地下室」に対応"),
    ("p06", "upbeat jazz with saxophone and piano",
     "既存プールにある方向"),
    ("p07", "baroque harpsichord piece",
     "クラシカル。注入でここへ流れる現象が起きていた"),
    ("p08", "energetic electronic dance beat with synth",
     "EDM。拍が明確"),
    ("p09", "lofi hip hop with mellow piano",
     "デモの定番。ループ前提のジャンル"),
    ("p10", "full orchestra with brass fanfare dramatic",
     "大編成。中間発表のお題に近い"),
    ("p11", "acoustic guitar fingerpicking warm",
     "弾き語り系。アタックが聴こえるか"),
    ("p12", "tense staccato strings suspenseful",
     "緊張感。短い音の連なり"),
]

# 同一プロンプトで x_T だけ変えたときのばらつき（第1段階のガチャに相当）
VARIATION_PROMPT = ("v", "aggressive electric guitar rock riff", 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--durations", type=float, nargs="+", default=[2.5, 10.0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import soundfile as sf
    from diffusers import MusicLDMPipeline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("モデルをロード中...")
    pipe = MusicLDMPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    sr = pipe.vocoder.config.sampling_rate
    ch = pipe.unet.config.in_channels
    print(f"  device={device} / sr={sr} / UNet in_channels={ch}")

    def encode(text: str) -> torch.Tensor:
        tok = pipe.tokenizer([text], padding=True, truncation=True,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            return pipe.text_encoder.get_text_features(**tok)

    def latent_shape(n: int, dur: float):
        return (n, ch, int(dur * sr / 160 / 4), 16)

    def generate(c: torch.Tensor, dur: float, seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        x_T = torch.randn(latent_shape(1, dur), generator=g, device=device)
        with torch.no_grad():
            out = pipe(prompt_embeds=c, latents=x_T,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance, audio_length_in_s=dur)
        return out.audios[0]

    rows = []
    t_all = time.time()

    # --- 12 プロンプト × 各音声長（seed はプロンプトごとに固定し、長さ間で揃える） ---
    for i, (pid, prompt, intent) in enumerate(PROMPTS):
        c = encode(prompt)
        seed = 1000 + i
        for dur in args.durations:
            name = f"{pid}_{dur:g}s.wav"
            t = time.time()
            audio = generate(c, dur, seed)
            sf.write(str(args.out_dir / name), audio, samplerate=sr)
            print(f"  {name:16s} {time.time() - t:5.1f}秒  {prompt}")
            rows.append({"file": name, "id": pid, "prompt": prompt,
                         "duration_s": dur, "seed": seed, "intent": intent,
                         "actual_s": round(len(audio) / sr, 2)})

    # --- 同一プロンプト・異なる x_T（第1段階のガチャ相当、2.5 秒のみ） ---
    vid, vprompt, vn = VARIATION_PROMPT
    c = encode(vprompt)
    for k in range(vn):
        name = f"{vid}{k + 1}_2.5s_xT.wav"
        seed = 7000 + k
        audio = generate(c, 2.5, seed)
        sf.write(str(args.out_dir / name), audio, samplerate=sr)
        print(f"  {name:16s}        {vprompt}（x_T 違い {k + 1}/{vn}）")
        rows.append({"file": name, "id": f"{vid}{k + 1}", "prompt": vprompt,
                     "duration_s": 2.5, "seed": seed,
                     "intent": f"同一プロンプト・x_T 違い {k + 1}/{vn}",
                     "actual_s": round(len(audio) / sr, 2)})

    # --- マニフェスト ---
    csv_path = args.out_dir / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = args.out_dir / "manifest.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# MusicLDM 試聴サンプル\n\n")
        f.write(f"- モデル: `{args.model}`\n")
        f.write(f"- DDIM {args.steps} ステップ / guidance {args.guidance} / "
                f"sampling rate {sr} Hz\n")
        f.write("- 同一プロンプトの 2.5 秒と 10 秒は **同じ seed** なので直接比較できる\n\n")
        f.write("## 判断すること\n\n")
        f.write("**MusicLDM の 2.5 秒は音楽として体裁を保っているか。**\n")
        f.write("AudioLDM（一般音響モデル）では「2.5 秒では音楽といえない」と指摘された。\n")
        f.write("音楽特化モデルで成立するなら、探索回数を確保したまま体裁を両立できる。\n\n")
        f.write("## 一覧\n\n")
        f.write("| ID | プロンプト | 狙い | 2.5 秒 | 10 秒 |\n")
        f.write("|---|---|---|---|---|\n")
        for pid, prompt, intent in PROMPTS:
            a = f"`{pid}_2.5s.wav`"
            b = f"`{pid}_10s.wav`" if 10.0 in args.durations else "—"
            f.write(f"| {pid} | `{prompt}` | {intent} | {a} | {b} |\n")
        f.write(f"\n## 同一プロンプト・異なる $x_T$（第1段階のガチャ相当・2.5 秒）\n\n")
        f.write(f"プロンプト: `{vprompt}`\n\n")
        for k in range(vn):
            f.write(f"- `{vid}{k + 1}_2.5s_xT.wav`\n")
        f.write("\n→ **2.5 秒でも音色・質感の差が聴き分けられるか**（聴き分けられなければ"
                "第1段階が成立しない）\n")

    print(f"\n✅ 完了 {time.time() - t_all:.1f}秒 / {len(rows)} ファイル")
    print(f"   音声: {args.out_dir}")
    print(f"   一覧: {md} / {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
