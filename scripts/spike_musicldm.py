"""MusicLDM 移行のスパイク：提案手法の前提が MusicLDM で成立するかを確認する。

本手法が生成モデルに要求するのは、条件ベクトル c と初期ノイズ x_T の2入力へ
**独立に介入できること**、および c が超球面上にあること（Slerp の前提）である。
diffusers の MusicLDMPipeline でそれが成立するかを、実際に音を出して確認する。

検証項目:
  1. prompt_embeds（= c）の形状と L2 ノルム
  2. latents（= x_T）を注入して音が出る／同じ latents で再現する
  3. 同じ c・異なる x_T → 音響テクスチャだけが変わる（第1段階の前提）
  4. Slerp した c で音が出る（第2段階の前提）
  5. 音声長の制約（AudioLDM は 2.5 秒の倍数のみ有効だった）
  6. 意味方向プール 1,407 方向を MusicLDM の CLAP で埋め込み直し、
     類似度分布から注入帯を再決定する

使用方法:
    NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \
    python scripts/spike_musicldm.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "ucsd-reach/musicldm"
POOL_JSON = "scripts/outputs/semantic_pool/semantic_pool.json"
OUT_DIR = Path("output/spike_musicldm")


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f" {title}")
    print("=" * 68)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=200, help="DDIM 相当のステップ数")
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--skip-pool", action="store_true")
    args = ap.parse_args()

    results: dict = {"model": args.model}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    section("0. モデルのロード")
    from diffusers import MusicLDMPipeline

    t0 = time.time()
    pipe = MusicLDMPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"  ロード完了 {time.time() - t0:.1f} 秒 / device={device}")
    print(f"  text_encoder : {type(pipe.text_encoder).__name__}")
    print(f"  unet         : {type(pipe.unet).__name__}")
    print(f"  vae          : {type(pipe.vae).__name__}")
    print(f"  vocoder      : {type(pipe.vocoder).__name__}")
    sr = pipe.vocoder.config.sampling_rate
    print(f"  sampling_rate: {sr}")
    results["sampling_rate"] = sr

    # ------------------------------------------------------------------
    section("1. prompt_embeds（= c）の形状と L2 ノルム")
    prompts = ["aggressive electric guitar", "mellow piano melody"]
    tok = pipe.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = pipe.text_encoder.get_text_features(**tok)
    norms = emb.norm(dim=-1)
    print(f"  shape = {tuple(emb.shape)}   ← AudioLDM 実装の (B,1,512) と異なり 2 次元")
    print(f"  L2 ノルム = {[round(float(n), 6) for n in norms]}")
    unit = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
    print(f"  単位ベクトルか: {'✅ はい（超球面上。Slerp の前提が成立）' if unit else '❌ いいえ'}")
    cos12 = float(torch.nn.functional.normalize(emb[0], dim=0)
                  @ torch.nn.functional.normalize(emb[1], dim=0))
    print(f"  2 方向間の cos = {cos12:.3f}")
    results["embed_shape"] = list(emb.shape)
    results["embed_unit_norm"] = unit

    def gen(prompt_embeds, latents, duration=None, tag="out"):
        """prompt_embeds と latents を与えて生成する（2入力への独立介入）。"""
        with torch.no_grad():
            out = pipe(
                prompt_embeds=prompt_embeds,
                latents=latents,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                audio_length_in_s=duration or args.duration,
            )
        return out.audios[0]

    # ------------------------------------------------------------------
    section("2. latents（= x_T）の注入と再現性")
    # まず latents の形状を pipeline に作らせて確認する
    with torch.no_grad():
        probe = pipe(prompt="test", num_inference_steps=2,
                     audio_length_in_s=args.duration, output_type="latent")
    lat_shape = tuple(probe.audios.shape) if hasattr(probe, "audios") else None
    print(f"  output_type='latent' の形状 = {lat_shape}")

    # vocoder/vae の設定から x_T の形状を組む
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    mel_bins = pipe.vocoder.config.model_in_dim
    # audio_length_in_s → mel フレーム数 → 潜在の時間軸
    hop = getattr(pipe.vocoder.config, "hop_size", None) or 160
    print(f"  vae_scale_factor={vae_scale} / mel_bins={mel_bins} / hop={hop}")

    g = torch.Generator(device=device).manual_seed(1234)
    shape = (1, pipe.unet.config.in_channels,
             int(args.duration * sr / hop / vae_scale),
             mel_bins // vae_scale)
    print(f"  組んだ x_T の形状 = {shape}")
    try:
        xT_a = torch.randn(shape, generator=g, device=device)
        c0 = emb[0:1]
        a1 = gen(c0, xT_a, tag="a1")
        print(f"  ✅ 生成できた: {a1.shape}（{a1.shape[-1] / sr:.2f} 秒）")
        save_wav(OUT_DIR / "01_xT-A_c0.wav", a1, sr)

        a2 = gen(c0, xT_a.clone(), tag="a2")
        same = float(np.abs(a1 - a2).max())
        print(f"  同一 x_T・同一 c での再現性: 最大差 {same:.2e} "
              f"{'✅ 決定論的' if same < 1e-3 else '⚠ 差あり'}")
        results["deterministic_max_diff"] = same
        results["latent_shape"] = list(shape)
        results["audio_seconds"] = a1.shape[-1] / sr
    except Exception as e:
        print(f"  ❌ 失敗: {type(e).__name__}: {e}")
        results["latent_injection_error"] = f"{type(e).__name__}: {e}"
        json.dump(results, open(OUT_DIR / "spike_result.json", "w"),
                  ensure_ascii=False, indent=2)
        return 1

    # ------------------------------------------------------------------
    section("3. 同じ c・異なる x_T（第1段階の前提）")
    g2 = torch.Generator(device=device).manual_seed(9999)
    xT_b = torch.randn(shape, generator=g2, device=device)
    b1 = gen(c0, xT_b, tag="b1")
    save_wav(OUT_DIR / "02_xT-B_c0.wav", b1, sr)
    diff_xT = float(np.abs(a1 - b1).mean())
    print(f"  x_T を変えたときの平均絶対差 = {diff_xT:.4f} "
          f"{'✅ 音が変わる' if diff_xT > 1e-3 else '❌ 変わらない'}")
    results["diff_by_xT"] = diff_xT

    section("4. 同じ x_T・異なる c（第2段階の前提）")
    c1 = emb[1:2]
    a3 = gen(c1, xT_a, tag="a3")
    save_wav(OUT_DIR / "03_xT-A_c1.wav", a3, sr)
    diff_c = float(np.abs(a1 - a3).mean())
    print(f"  c を変えたときの平均絶対差 = {diff_c:.4f} "
          f"{'✅ 音が変わる' if diff_c > 1e-3 else '❌ 変わらない'}")
    results["diff_by_c"] = diff_c

    # Slerp した c
    from audioldm.iec import slerp

    c_mid = slerp(c0.flatten(), c1.flatten(), 0.5).view_as(c0)
    print(f"  Slerp 中点の L2 ノルム = {float(c_mid.norm()):.6f}")
    a4 = gen(c_mid, xT_a, tag="slerp")
    save_wav(OUT_DIR / "04_xT-A_slerp.wav", a4, sr)
    print(f"  ✅ Slerp した c で生成できた")
    results["slerp_norm"] = float(c_mid.norm())

    # ------------------------------------------------------------------
    section("5. 音声長の制約")
    for d in (2.5, 5.0, 10.0):
        try:
            gd = torch.Generator(device=device).manual_seed(7)
            sh = (1, pipe.unet.config.in_channels,
                  int(d * sr / hop / vae_scale), mel_bins // vae_scale)
            x = torch.randn(sh, generator=gd, device=device)
            t = time.time()
            au = gen(c0, x, duration=d)
            print(f"  {d:4.1f} 秒 → ✅ {au.shape[-1] / sr:.2f} 秒 / "
                  f"生成 {time.time() - t:.1f} 秒 / x_T 形状 {sh}")
            save_wav(OUT_DIR / f"05_dur{d:.1f}s.wav", au, sr)
            results[f"duration_{d}"] = {"ok": True, "sec": au.shape[-1] / sr}
        except Exception as e:
            print(f"  {d:4.1f} 秒 → ❌ {type(e).__name__}: {e}")
            results[f"duration_{d}"] = {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    if not args.skip_pool and os.path.exists(POOL_JSON):
        section("6. 意味方向プールを MusicLDM の CLAP で埋め込み直す")
        from audioldm.prompt_pool import load_pool_json

        terms = load_pool_json(POOL_JSON)
        print(f"  {len(terms)} 方向をエンコード中...")
        t = time.time()
        embs = []
        B = 64
        with torch.no_grad():
            for i in range(0, len(terms), B):
                tk = pipe.tokenizer(terms[i:i + B], padding=True,
                                    truncation=True, return_tensors="pt").to(device)
                embs.append(pipe.text_encoder.get_text_features(**tk).cpu())
        E = torch.cat(embs, dim=0)
        print(f"  完了 {time.time() - t:.1f} 秒 / shape={tuple(E.shape)}")

        out_pt = Path(POOL_JSON.replace(".json", "_embeddings_musicldm.pt"))
        torch.save({"terms": terms, "embeddings": E, "model_name": args.model}, out_pt)
        print(f"  保存: {out_pt}")

        En = torch.nn.functional.normalize(E.float(), dim=-1)
        S = En @ En.T
        rng = np.random.default_rng(0)
        idx = rng.choice(len(terms), size=40, replace=False)
        vals = []
        for i in idx:
            s = S[i].tolist()
            vals.extend([v for j, v in enumerate(s) if j != i])
        vals.sort()

        def q(p):
            return vals[int(p * (len(vals) - 1))]

        print(f"\n  c* 40 個 × {len(terms) - 1} 方向 (n={len(vals):,}) の cos 分布")
        pct = {}
        for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            pct[f"{int(p * 100)}%"] = round(q(p), 3)
            print(f"    {int(p * 100):3d}%tile: {q(p):.3f}")
        print(f"    min={vals[0]:.3f} max={vals[-1]:.3f} mean={statistics.mean(vals):.3f}")
        results["pool_similarity_percentiles"] = pct
        results["pool_similarity_mean"] = round(statistics.mean(vals), 4)

        print("\n  注入帯の候補（AudioLDM では [0.50, 0.80] を採用していた）")
        for lo, hi in [(q(0.50), q(0.90)), (q(0.60), q(0.92)), (0.50, 0.80)]:
            n = sum(1 for v in vals if lo <= v <= hi)
            print(f"    [{lo:.2f}, {hi:.2f}] → 平均 {n / len(idx):.0f} 方向/c* "
                  f"({n / len(vals) * 100:.0f}%)")
        print(f"\n  → 中央値〜90%tile = [{q(0.50):.2f}, {q(0.90):.2f}] を暫定の帯とする案")
        results["suggested_band"] = [round(q(0.50), 2), round(q(0.90), 2)]

    json.dump(results, open(OUT_DIR / "spike_result.json", "w"),
              ensure_ascii=False, indent=2)
    section("結論")
    print(f"  2 入力への独立介入: "
          f"{'✅ 成立' if results.get('diff_by_xT', 0) > 1e-3 and results.get('diff_by_c', 0) > 1e-3 else '❌'}")
    print(f"  Slerp の前提（超球面）: {'✅ 成立' if results.get('embed_unit_norm') else '❌'}")
    print(f"  音声は {OUT_DIR} / 結果は {OUT_DIR}/spike_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
