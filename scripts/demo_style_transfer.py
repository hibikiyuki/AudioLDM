#!/usr/bin/env python3
"""
スタイル転送 WAV 生成デモ

音声A（コンテンツ源）と音声B（スタイル参照）を与えると、
CLAPでスタイル語を選出し、SDEditで「AのコンテンツにBのスタイル」を転送した
音声を世代別に保存する。

Usage:
    # WAVファイルを指定
    NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/demo_style_transfer.py \
        --audio_a content.wav --audio_b style.wav

    # テキストプロンプトから自動生成して使用
    NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/demo_style_transfer.py \
        --prompt_a "acoustic guitar melody" \
        --prompt_b "jazz piano improvisation"

    # 進化ステップ数・個体数を調整
    NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/demo_style_transfer.py \
        --audio_a content.wav --audio_b style.wav \
        --pop_size 6 --evolve_steps 3 --noise_min 0.1 --noise_max 0.5
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, ".")

SAMPLE_RATE = 16000


# ─────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────

def save_wav(path: str, wav: np.ndarray) -> float:
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        wav = wav[0]
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    sf.write(path, wav, SAMPLE_RATE)
    return rms


def header(text: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def generate_from_prompt(pipe, prompt: str, guidance_scale: float = 7.5) -> np.ndarray:
    """テキストプロンプトから音声波形を生成する。"""
    from audioldm.latent_diffusion.ddim import DDIMSampler

    with pipe.latent_diffusion.ema_scope("TextToAudio"):
        with torch.no_grad():
            sampler = DDIMSampler(pipe.latent_diffusion)
            sampler.make_schedule(
                ddim_num_steps=pipe.ddim_steps, ddim_eta=0.0, verbose=False)

            c = pipe.latent_diffusion.cond_stage_model([prompt, prompt])[0:1]
            uc = pipe.latent_diffusion.cond_stage_model.get_unconditional_condition(1)

            z0, _ = sampler.sample(
                S=pipe.ddim_steps,
                conditioning=c,
                batch_size=1,
                shape=list(pipe.latent_shape),
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
                verbose=False,
            )
    return pipe._decode_z0_to_audio(z0)


def wav_to_tmp(wav: np.ndarray, prefix: str = "tmp") -> str:
    """波形を一時 WAV ファイルに保存してパスを返す。"""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix + "_")
    os.close(fd)
    save_wav(path, wav)
    return path


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="スタイル転送 WAV 生成デモ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 入力ソース
    src = parser.add_argument_group("入力ソース (WAVまたはプロンプトを指定)")
    src.add_argument("--audio_a", help="コンテンツ源 WAV ファイル")
    src.add_argument("--audio_b", help="スタイル参照 WAV ファイル")
    src.add_argument("--prompt_a", default="acoustic guitar melody",
                     help="音声Aのテキストプロンプト (--audio_a 未指定時に使用)")
    src.add_argument("--prompt_b", default="jazz piano improvisation",
                     help="音声Bのテキストプロンプト (--audio_b 未指定時に使用)")

    # スタイル転送パラメータ
    st = parser.add_argument_group("スタイル転送パラメータ")
    st.add_argument("--base_prompt", default="",
                    help="スタイル語に付加するベースプロンプト")
    st.add_argument("--top_k", type=int, default=5,
                    help="CLAPが選ぶスタイル語の数 (デフォルト: 5)")
    st.add_argument("--noise_min", type=float, default=0.15,
                    help="noise_strength の最小値 (デフォルト: 0.15)")
    st.add_argument("--noise_max", type=float, default=0.45,
                    help="noise_strength の最大値 (デフォルト: 0.45)")
    st.add_argument("--gs_min", type=float, default=3.0,
                    help="guidance_scale の最小値 (デフォルト: 3.0)")
    st.add_argument("--gs_max", type=float, default=10.0,
                    help="guidance_scale の最大値 (デフォルト: 10.0)")

    # 進化パラメータ
    ev = parser.add_argument_group("進化パラメータ")
    ev.add_argument("--pop_size", type=int, default=4,
                    help="個体数 (デフォルト: 4)")
    ev.add_argument("--evolve_steps", type=int, default=2,
                    help="進化ステップ数 (デフォルト: 2)")
    ev.add_argument("--select_best", type=int, default=2,
                    help="各世代で親として使う個体数 (デフォルト: 2、先頭から)")
    ev.add_argument("--elite_count", type=int, default=1,
                    help="エリート保存数 (デフォルト: 1)")

    # モデル・出力
    parser.add_argument("--model", default="audioldm-s-full-v2",
                        help="AudioLDMモデル名 (デフォルト: audioldm-s-full-v2)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="音声長(秒) (デフォルト: 5.0)")
    parser.add_argument("--ddim_steps", type=int, default=200,
                        help="DDIM ステップ数 (デフォルト: 200)")
    parser.add_argument("--out_dir", default="output/style_transfer_demo",
                        help="出力ディレクトリ (デフォルト: output/style_transfer_demo)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ─────────────────────────────────────────────────────
    # モデルロード
    # ─────────────────────────────────────────────────────
    header("モデルロード")
    from audioldm.iec_pipeline import AudioLDM_IEC
    pipe = AudioLDM_IEC(
        model_name=args.model,
        ga_mode="latent",
        duration=args.duration,
        ddim_steps=args.ddim_steps,
        population_size=args.pop_size,
    )
    print(f"  モデル   : {args.model}")
    print(f"  個体数   : {args.pop_size}")
    print(f"  音声長   : {args.duration}s")
    print(f"  DDIMステップ: {args.ddim_steps}")

    # ─────────────────────────────────────────────────────
    # 入力音声の準備
    # ─────────────────────────────────────────────────────
    header("入力音声の準備")
    tmp_files = []

    if args.audio_a:
        audio_a_path = args.audio_a
        print(f"  音声A (コンテンツ): {audio_a_path}")
    else:
        print(f"  音声Aをプロンプトから生成中: \"{args.prompt_a}\"")
        wav_a = generate_from_prompt(pipe, args.prompt_a)
        audio_a_path = wav_to_tmp(wav_a, "audio_a")
        tmp_files.append(audio_a_path)
        rms_a = save_wav(os.path.join(args.out_dir, "original_a.wav"), wav_a)
        print(f"  → original_a.wav  rms={rms_a:.4f}")

    if args.audio_b:
        audio_b_path = args.audio_b
        print(f"  音声B (スタイル参照): {audio_b_path}")
    else:
        print(f"  音声Bをプロンプトから生成中: \"{args.prompt_b}\"")
        wav_b = generate_from_prompt(pipe, args.prompt_b)
        audio_b_path = wav_to_tmp(wav_b, "audio_b")
        tmp_files.append(audio_b_path)
        rms_b = save_wav(os.path.join(args.out_dir, "original_b.wav"), wav_b)
        print(f"  → original_b.wav  rms={rms_b:.4f}")

    # 入力ファイルをコピーして保存
    for src_path, dst_name in [(audio_a_path, "original_a.wav"),
                                (audio_b_path, "original_b.wav")]:
        dst = os.path.join(args.out_dir, dst_name)
        if not os.path.exists(dst):
            wav_src, sr = sf.read(src_path, always_2d=False)
            sf.write(dst, wav_src, sr)
            print(f"  コピー: {dst_name}  sr={sr}")

    # ─────────────────────────────────────────────────────
    # 第0世代: 初期化
    # ─────────────────────────────────────────────────────
    header("第0世代: 初期化 (CLAPスタイル語ランキング + SDEdit)")
    results, ranked_words = pipe.initialize_style_transfer_population(
        audio_a_path=audio_a_path,
        audio_b_path=audio_b_path,
        base_prompt=args.base_prompt,
        top_k_styles=args.top_k,
        noise_strength_range=(args.noise_min, args.noise_max),
        guidance_scale_range=(args.gs_min, args.gs_max),
    )

    # スタイル語を保存・表示
    style_words_path = os.path.join(args.out_dir, "style_words.txt")
    with open(style_words_path, "w", encoding="utf-8") as f:
        f.write(f"スタイル参照: {audio_b_path}\n\n")
        f.write("CLAPスコア上位スタイル語:\n")
        for word, score in ranked_words:
            f.write(f"  {score:+.4f}  {word}\n")

    print(f"\n  CLAPスタイル語 top-{args.top_k}:")
    for word, score in ranked_words:
        print(f"    {score:+.4f}  {word}")

    gen0_dir = os.path.join(args.out_dir, "gen0")
    os.makedirs(gen0_dir, exist_ok=True)

    print(f"\n  生成結果 (gen0):")
    gen_summary = []
    for i, (g, wav) in enumerate(results):
        fname = f"ind{i}_ns{g.noise_strength:.2f}_gs{g.guidance_scale:.1f}.wav"
        fpath = os.path.join(gen0_dir, fname)
        rms = save_wav(fpath, wav)
        gen_summary.append((0, i, g, fname, rms))
        print(f"    ind{i}: noise_strength={g.noise_strength:.3f}  "
              f"guidance_scale={g.guidance_scale:.2f}  rms={rms:.4f}  → {fname}")

    # ─────────────────────────────────────────────────────
    # 進化ステップ
    # ─────────────────────────────────────────────────────
    for step in range(1, args.evolve_steps + 1):
        header(f"第{step}世代: 進化 (先頭{args.select_best}個体を親に選択)")

        selected = list(range(min(args.select_best, args.pop_size)))
        results2 = pipe.evolve_style_transfer_population(
            selected_indices=selected,
            elite_count=args.elite_count,
        )

        gen_dir = os.path.join(args.out_dir, f"gen{step}")
        os.makedirs(gen_dir, exist_ok=True)

        print(f"\n  生成結果 (gen{step}):")
        for i, (g, wav) in enumerate(results2):
            is_elite = g.metadata.get("elite", False)
            elite_tag = "_elite" if is_elite else ""
            fname = (f"ind{i}{elite_tag}_ns{g.noise_strength:.2f}"
                     f"_gs{g.guidance_scale:.1f}"
                     f"_ms{g.mask_start:.2f}-{g.mask_end:.2f}.wav")
            fpath = os.path.join(gen_dir, fname)
            rms = save_wav(fpath, wav)
            gen_summary.append((step, i, g, fname, rms))
            print(f"    ind{i}: ns={g.noise_strength:.3f}  "
                  f"gs={g.guidance_scale:.2f}  "
                  f"mask=[{g.mask_start:.2f},{g.mask_end:.2f}]  "
                  f"elite={is_elite}  rms={rms:.4f}")

        results = results2  # 次世代の親候補を更新

    # ─────────────────────────────────────────────────────
    # サマリー出力
    # ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    header(f"完了  ({elapsed:.1f}秒)")

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"スタイル転送デモ サマリー\n")
        f.write(f"{'='*60}\n")
        f.write(f"モデル     : {args.model}\n")
        f.write(f"音声A      : {audio_a_path}\n")
        f.write(f"音声B      : {audio_b_path}\n")
        f.write(f"ベースプロンプト: {args.base_prompt or '(なし)'}\n")
        f.write(f"スタイルプロンプト: {results[0][0].style_prompt if results else ''}\n\n")
        f.write(f"CLAPスタイル語 top-{args.top_k}:\n")
        for word, score in ranked_words:
            f.write(f"  {score:+.4f}  {word}\n")
        f.write(f"\n{'─'*60}\n")
        f.write(f"{'世代':>4}  {'個体':>4}  {'noise':>6}  {'gs':>6}  "
                f"{'mask_start':>10}  {'mask_end':>8}  {'rms':>7}  ファイル名\n")
        f.write(f"{'─'*60}\n")
        for gen_n, ind_n, g, fname, rms in gen_summary:
            f.write(f"{gen_n:>4}  {ind_n:>4}  {g.noise_strength:>6.3f}  "
                    f"{g.guidance_scale:>6.2f}  {g.mask_start:>10.3f}  "
                    f"{g.mask_end:>8.3f}  {rms:>7.4f}  {fname}\n")

    print(f"  出力先: {os.path.abspath(args.out_dir)}/")
    print(f"  ファイル構成:")
    print(f"    original_a.wav       ← コンテンツ音源")
    print(f"    original_b.wav       ← スタイル参照音源")
    print(f"    style_words.txt      ← CLAPスタイル語スコア")
    print(f"    gen0/                ← 初期世代 ({args.pop_size}個体)")
    for step in range(1, args.evolve_steps + 1):
        print(f"    gen{step}/                ← 進化第{step}世代 ({args.pop_size}個体)")
    print(f"    summary.txt          ← 全個体の遺伝子・RMSサマリー")

    # 一時ファイルの削除
    for p in tmp_files:
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"\n  生成ファイル数: {len(gen_summary)} 個")
    print(f"  経過時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
