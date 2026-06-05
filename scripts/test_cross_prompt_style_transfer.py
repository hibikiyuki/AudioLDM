#!/usr/bin/env python3
"""
IEC クロスプロンプト スタイル転送 検証スクリプト (改訂版)

旧版 test_iec_style_transfer_verification.py の3つの欠陥を修正する:

  【欠陥1】ベースライン依存・天井効果
    旧: style_gain = sim(out,B) - sim(A,B)
    → 同一プロンプト由来で sim(A,B) が既に高いと gain の上限が小さくなる
    新: style_ratio = sim(out,B) / sim(out,A)
    → Bへの傾き度を比率で評価。天井効果に依存しない

  【欠陥2】同一プロンプトによる知覚的限界
    旧: 同一プロンプトから A, B を生成 → 元々同クラスタで差が取れない
    新: 異種プロンプトペア (prompt_a ≠ prompt_b) を使い
        sim(A,B) < 0.4 を実験条件として統制する

  【欠陥3】コンテンツ保持の未測定
    旧: スタイル転送の効果しか測らない
    新: content_pres = chroma_sim(out,A) / chroma_sim(out,B)
        クロマグラム類似度で「メロディ構造が A から引き継がれているか」を独立測定

評価二軸:
  X軸: style_ratio  = sim(out,B) / sim(out,A)     [> 1.0 でスタイルが B 寄り]
  Y軸: content_pres = chroma_sim(out,A) / chroma_sim(out,B)  [> 1.0 でA構造維持]
  理想点: 両方が 1.0 を超える noise_strength のスイートスポットを探す

Usage:
    # Section 0 のみ (モデル不要)
    python scripts/test_cross_prompt_style_transfer.py

    # 統合テスト (モデル・GPU 必要)
    python scripts/test_cross_prompt_style_transfer.py --integration \\
        --model audioldm-s-full \\
        --prompt-a "quiet piano melody" \\
        --prompt-b "intense electronic music with heavy bass" \\
        --duration 5.0
"""

import sys
import os
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import librosa

sys.path.insert(0, ".")

# ─────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0


def ok(name: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {name}")


def fail(name: str, msg: str = ""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {name}" + (f": {msg}" if msg else ""))


def check(name: str, cond: bool, msg: str = ""):
    if cond:
        ok(name)
    else:
        fail(name, msg or "条件を満たしていない")


def info(msg: str):
    print(f"  [INFO] {msg}")


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────
# 指標計算
# ─────────────────────────────────────────

def style_ratio(sim_out_b: float, sim_out_a: float) -> float:
    """
    スタイル比率: sim(out,B) / sim(out,A)
    > 1.0  → 出力が A より B に近い（スタイル転送成功）
    = 1.0  → A と B への距離が等しい
    < 1.0  → A の方に近い（転送効果なし）
    分母がゼロに近い場合は 0.0 を返す。
    """
    if abs(sim_out_a) < 1e-8:
        return 0.0
    return sim_out_b / sim_out_a


def style_gain(sim_out_b: float, sim_ab: float) -> float:
    """旧指標: sim(out,B) - sim(A,B)。参考値として残す。"""
    return sim_out_b - sim_ab


def chroma_cosine_similarity(path_a: str, path_b: str, sr: int = 16000) -> float:
    """
    2つの音声ファイルのクロマグラム（平均ベクトル）のコサイン類似度を返す。

    クロマグラムは 12 音高クラスのエネルギー分布を捉えるため、
    メロディ・ハーモニー構造の類似度の代理指標として使う。
    CLAPが「全体的な雰囲気」を捉えるのに対し、こちらは「音程構造」に特化する。
    戻り値: 0.0〜1.0 (音声特徴量は非負なので負にはならない)
    """
    wav_a, _ = librosa.load(path_a, sr=sr, mono=True)
    wav_b, _ = librosa.load(path_b, sr=sr, mono=True)
    mean_a = librosa.feature.chroma_cqt(y=wav_a, sr=sr).mean(axis=1)  # (12,)
    mean_b = librosa.feature.chroma_cqt(y=wav_b, sr=sr).mean(axis=1)  # (12,)
    denom = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(mean_a, mean_b) / denom)


def content_preservation_ratio(
    chroma_sim_out_a: float, chroma_sim_out_b: float
) -> float:
    """
    コンテンツ保持比率: chroma_sim(out,A) / chroma_sim(out,B)
    > 1.0 → 出力が B よりも A の音程構造を引き継いでいる（コンテンツ保持）
    = 1.0 → A と B への構造類似度が等しい
    < 1.0 → B の構造に引っ張られている（コンテンツ破壊）
    """
    if abs(chroma_sim_out_b) < 1e-8:
        return 0.0
    return chroma_sim_out_a / chroma_sim_out_b


def clap_audio_embedding(pipeline, audio_path: str) -> torch.Tensor:
    """音声ファイルの CLAP audio embedding (1, 512) を返す。"""
    waveform = pipeline._load_waveform_for_clap(audio_path).to(pipeline.device)
    cond = pipeline.latent_diffusion.cond_stage_model
    orig_prob = cond.unconditional_prob
    cond.unconditional_prob = 0.0
    try:
        with torch.no_grad():
            cond.embed_mode = "audio"
            emb = cond(waveform).squeeze(1)  # (1, 512)
    finally:
        cond.embed_mode = "text"
        cond.unconditional_prob = orig_prob
    return emb


# ─────────────────────────────────────────
# Section 0: 単体テスト（モデル不要）
# ─────────────────────────────────────────

def run_section0():
    section("Section 0: 単体テスト (モデル不要)")

    # ────────────────────────────────────────────────────────
    # 0-A: 旧指標 style_gain の天井効果を数値で確認
    # ────────────────────────────────────────────────────────
    print("\n  [0-A] 旧指標(style_gain)の天井効果デモ")

    # シナリオ1: 同一プロンプト由来（高ベースライン）
    sim_ab_same  = 0.85   # 元々近い
    sim_out_b_s1 = 0.88   # わずかに改善
    sim_out_a_s1 = 0.82

    # シナリオ2: 異種プロンプト（低ベースライン）
    sim_ab_cross = 0.35   # 元々遠い
    sim_out_b_s2 = 0.65   # 大幅改善
    sim_out_a_s2 = 0.40

    gain_same  = style_gain(sim_out_b_s1, sim_ab_same)   # +0.03
    gain_cross = style_gain(sim_out_b_s2, sim_ab_cross)  # +0.30
    ratio_same  = style_ratio(sim_out_b_s1, sim_out_a_s1)  # 1.073
    ratio_cross = style_ratio(sim_out_b_s2, sim_out_a_s2)  # 1.625

    info(f"同一プロンプト: sim(A,B)={sim_ab_same:.2f}, sim(out,B)={sim_out_b_s1:.2f}")
    info(f"  style_gain  = {gain_same:+.3f}  ← 小さく見える（天井効果）")
    info(f"  style_ratio = {ratio_same:.3f}  ← Bへの傾き比率")
    info(f"異種プロンプト: sim(A,B)={sim_ab_cross:.2f}, sim(out,B)={sim_out_b_s2:.2f}")
    info(f"  style_gain  = {gain_cross:+.3f}  ← 大きく見える（低ベースラインの恩恵）")
    info(f"  style_ratio = {ratio_cross:.3f}  ← より明確な転送効果")

    # 天井効果: gain では同一プロンプトが不当に低くなる
    check(
        "0-A-1: 天井効果 – 異種プロンプトの gain が同一プロンプトの 5倍以上",
        gain_cross / max(abs(gain_same), 1e-8) > 5.0,
        f"gain_cross={gain_cross:.3f}, gain_same={gain_same:.3f}",
    )
    # ratio では両シナリオとも > 1.0 で正しく転送効果を捉えている
    check(
        "0-A-2: ratio – 同一プロンプトでも > 1.0 で転送効果を捉えられる",
        ratio_same > 1.0,
        f"ratio_same={ratio_same:.3f}",
    )
    check(
        "0-A-3: ratio – 異種プロンプトで明確な転送効果 (> 1.2)",
        ratio_cross > 1.2,
        f"ratio_cross={ratio_cross:.3f}",
    )

    # ────────────────────────────────────────────────────────
    # 0-B: style_ratio の単調性検証
    # ────────────────────────────────────────────────────────
    print("\n  [0-B] style_ratio の単調性検証")

    sim_out_a_fixed = 0.50
    # sim(out,B) が上がるほど style_ratio も上がるべき
    b_values = [0.30, 0.45, 0.50, 0.60, 0.70]
    ratios = [style_ratio(b, sim_out_a_fixed) for b in b_values]
    info("sim(out,B) が上がるにつれて style_ratio が単調増加するか:")
    for b, r in zip(b_values, ratios):
        info(f"  sim(out,B)={b:.2f} → style_ratio={r:.3f}")
    is_monotone = all(ratios[i] < ratios[i+1] for i in range(len(ratios)-1))
    check("0-B-1: style_ratio が sim(out,B) について単調増加", is_monotone)

    # sim(out,A) が上がるほど style_ratio は下がるべき
    a_values = [0.30, 0.40, 0.50, 0.60, 0.70]
    ratios_a = [style_ratio(0.55, a) for a in a_values]
    info("sim(out,A) が上がるにつれて style_ratio が単調減少するか:")
    for a, r in zip(a_values, ratios_a):
        info(f"  sim(out,A)={a:.2f} → style_ratio={r:.3f}")
    is_monotone_a = all(ratios_a[i] > ratios_a[i+1] for i in range(len(ratios_a)-1))
    check("0-B-2: style_ratio が sim(out,A) について単調減少", is_monotone_a)

    # ────────────────────────────────────────────────────────
    # 0-C: content_preservation_ratio の検証
    # ────────────────────────────────────────────────────────
    print("\n  [0-C] content_preservation_ratio の検証")

    # ケース1: 出力が A の構造を保持（理想的な転送）
    cp_ideal = content_preservation_ratio(chroma_sim_out_a=0.80, chroma_sim_out_b=0.30)
    check(
        "0-C-1: A構造保持時は content_pres > 1.0",
        cp_ideal > 1.0,
        f"content_pres={cp_ideal:.3f}",
    )

    # ケース2: 出力が B の構造に引っ張られた（コンテンツ破壊）
    cp_broken = content_preservation_ratio(chroma_sim_out_a=0.30, chroma_sim_out_b=0.75)
    check(
        "0-C-2: B構造に引っ張られた場合 content_pres < 1.0",
        cp_broken < 1.0,
        f"content_pres={cp_broken:.3f}",
    )

    # ケース3: 等距離のとき content_pres ≒ 1.0
    cp_equal = content_preservation_ratio(chroma_sim_out_a=0.50, chroma_sim_out_b=0.50)
    check(
        "0-C-3: A=B 距離のとき content_pres ≒ 1.0",
        abs(cp_equal - 1.0) < 1e-6,
        f"content_pres={cp_equal:.6f}",
    )

    info(f"理想的転送: {cp_ideal:.3f}  コンテンツ破壊: {cp_broken:.3f}  等距離: {cp_equal:.3f}")

    # ────────────────────────────────────────────────────────
    # 0-D: クロマグラム類似度の合成ベクトル検証
    # ────────────────────────────────────────────────────────
    print("\n  [0-D] クロマグラム類似度 (合成ベクトル)")

    # 同一ベクトルはコサイン類似度 1.0
    vec = np.array([0.5, 0.3, 0.1, 0.7, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.1, 0.8])
    denom = np.linalg.norm(vec) ** 2
    cos_identical = float(np.dot(vec, vec) / denom)
    check(
        "0-D-1: 同一クロマベクトルのコサイン類似度 = 1.0",
        abs(cos_identical - 1.0) < 1e-6,
        f"cos={cos_identical:.6f}",
    )

    # 直交ベクトルはコサイン類似度 0.0
    vec_orth = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0])
    # vec は偶数インデックスに非ゼロ、vec_orth は奇数インデックスに非ゼロ
    vec_even = np.array([0.5, 0.0, 0.3, 0.0, 0.7, 0.0, 0.4, 0.0, 0.2, 0.0, 0.6, 0.0])
    vec_odd  = np.array([0.0, 0.4, 0.0, 0.8, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.0, 0.2])
    cos_orth = float(np.dot(vec_even, vec_odd) / (np.linalg.norm(vec_even) * np.linalg.norm(vec_odd)))
    check(
        "0-D-2: 直交クロマベクトルのコサイン類似度 = 0.0",
        abs(cos_orth) < 1e-6,
        f"cos={cos_orth:.6f}",
    )

    # ────────────────────────────────────────────────────────
    # 0-E: 二軸評価の領域分類
    # ────────────────────────────────────────────────────────
    print("\n  [0-E] 二軸評価の領域分類")

    def classify_result(sr: float, cp: float) -> str:
        """style_ratio と content_pres から領域を分類する。"""
        style_ok   = sr > 1.0
        content_ok = cp > 1.0
        if style_ok and content_ok:
            return "IDEAL"       # 両軸クリア (理想的なスタイル転送)
        elif style_ok:
            return "STYLE_ONLY"  # スタイルのみ獲得（コンテンツ破壊）
        elif content_ok:
            return "CONTENT_ONLY"  # コンテンツは保持だがスタイル未転送
        else:
            return "FAILED"     # 両方ダメ

    cases = [
        (1.3, 1.2, "IDEAL"),
        (1.3, 0.8, "STYLE_ONLY"),
        (0.8, 1.2, "CONTENT_ONLY"),
        (0.8, 0.8, "FAILED"),
    ]
    all_ok = True
    for sr_, cp_, expected in cases:
        result = classify_result(sr_, cp_)
        correct = result == expected
        all_ok = all_ok and correct
        info(f"  ratio={sr_:.1f}, content_pres={cp_:.1f} → {result} (期待: {expected}) {'OK' if correct else 'NG'}")
    check("0-E-1: 二軸領域分類が正確", all_ok)


# ─────────────────────────────────────────
# Section 1〜4: 統合テスト
# ─────────────────────────────────────────

def _generate_and_save(pipeline, prompt: str, tmpdir: str, tag: str) -> str:
    """プロンプトから 1 体を z0 モードで生成して wav として保存し、パスを返す。"""
    import soundfile as sf
    results = pipeline.initialize_population_z0(prompt=prompt)
    # 最初の個体だけ使う
    _, waveform = results[0]
    path = os.path.join(tmpdir, f"{tag}.wav")
    sf.write(path, np.array(waveform).squeeze(), 16000)
    return path


def run_integration(args):
    from audioldm.iec_pipeline import AudioLDM_IEC
    from audioldm.iec import StyleTransferGenotype
    import soundfile as sf

    section("統合テスト準備: モデルロード")
    pipeline = AudioLDM_IEC(
        model_name=args.model,
        population_size=1,  # 多様性検証は不要。1体ずつ生成する
        duration=args.duration,
    )
    info(f"モデル: {args.model}, 時間: {args.duration}s")

    tmpdir = tempfile.mkdtemp(prefix="iec_cross_prompt_")
    info(f"作業ディレクトリ: {tmpdir}")

    # ─── Section 1: クロスプロンプトペアの検証 ────────────────────────
    section("Section 1: クロスプロンプト ペア検証")
    info(f"Parent A プロンプト: '{args.prompt_a}'")
    info(f"Parent B プロンプト: '{args.prompt_b}'")

    path_a = _generate_and_save(pipeline, args.prompt_a, tmpdir, "parent_A")
    path_b = _generate_and_save(pipeline, args.prompt_b, tmpdir, "parent_B")
    info(f"生成完了: {path_a}, {path_b}")

    emb_a = clap_audio_embedding(pipeline, path_a)
    emb_b = clap_audio_embedding(pipeline, path_b)
    sim_ab = F.cosine_similarity(emb_a, emb_b, dim=1).item()
    info(f"sim(A, B) = {sim_ab:.4f}")

    # 1-1: ベースラインが低いことを確認 (実験の前提条件)
    check(
        "1-1: sim(A,B) < 0.5 (クロスプロンプト条件が成立)",
        sim_ab < 0.5,
        f"sim(A,B)={sim_ab:.4f}。プロンプトを更に対照的にすることを検討してください。",
    )

    chroma_ab = chroma_cosine_similarity(path_a, path_b)
    info(f"chroma_sim(A, B) = {chroma_ab:.4f} (コンテンツ構造の参照値)")

    # ─── Section 2: 差分スタイル語の抽出 ─────────────────────────────
    section("Section 2: 差分スタイル語の抽出")

    std_ranked  = pipeline.rank_style_words(path_b, top_k=5)
    diff_ranked = pipeline.rank_differential_style_words(path_b, path_a, top_k=5, pool_size=10)

    std_words  = [w for w, _ in std_ranked]
    diff_words = [w for w, _ in diff_ranked]
    diff_only  = set(diff_words) - set(std_words)

    info(f"標準 Top-5 (B): {std_words}")
    info(f"差分 Top-5 (B-A集合差分, pool=10): {diff_words}")
    info(f"差分語のみ登場: {diff_only}")

    check(
        "2-1: 差分語に標準語と異なる語が含まれる",
        len(diff_only) > 0,
        f"diff_only={diff_only}",
    )

    # ─── Section 3: 二軸評価 ──────────────────────────────────────────
    section("Section 3: スタイル転送 × コンテンツ保持 二軸評価")

    z0_content = pipeline.encode_audio_to_z0(path_a)

    std_prompt  = pipeline.build_style_prompt(args.prompt_a, std_ranked)
    diff_prompt = pipeline.build_style_prompt(args.prompt_a, list(diff_ranked))
    info(f"標準スタイルプロンプト: {std_prompt}")
    info(f"差分スタイルプロンプト: {diff_prompt}")

    noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  {'ns':>4}  {'prompt':>12}  {'sim(out,A)':>10}  {'sim(out,B)':>10}  "
          f"{'style_ratio':>11}  {'style_gain':>10}  {'chroma(out,A)':>13}  "
          f"{'chroma(out,B)':>13}  {'content_pres':>12}  {'区分':>12}")
    print("  " + "─" * 115)

    results_table = []
    for prompt_label, style_prompt in [("standard", std_prompt), ("differential", diff_prompt)]:
        for ns in noise_strengths:
            g = StyleTransferGenotype(
                z0_content=z0_content,
                style_prompt=style_prompt,
                noise_strength=ns,
                guidance_scale=args.guidance_scale,
                seed=42,
            )
            waveforms = pipeline.generate_style_transfer_audio_batch([g])
            out_path = os.path.join(tmpdir, f"out_{prompt_label}_ns{ns:.1f}.wav")
            sf.write(out_path, np.array(waveforms[0]).squeeze(), 16000)

            emb_out = clap_audio_embedding(pipeline, out_path)
            sim_out_a = F.cosine_similarity(emb_out, emb_a, dim=1).item()
            sim_out_b = F.cosine_similarity(emb_out, emb_b, dim=1).item()

            sr    = style_ratio(sim_out_b, sim_out_a)
            sg    = style_gain(sim_out_b, sim_ab)
            ca    = chroma_cosine_similarity(out_path, path_a)
            cb    = chroma_cosine_similarity(out_path, path_b)
            cp    = content_preservation_ratio(ca, cb)

            style_ok   = sr > 1.0
            content_ok = cp > 1.0
            if style_ok and content_ok:
                label = "IDEAL"
            elif style_ok:
                label = "STYLE_ONLY"
            elif content_ok:
                label = "CONTENT_ONLY"
            else:
                label = "FAILED"

            results_table.append({
                "prompt_label": prompt_label,
                "ns": ns,
                "sim_out_a": sim_out_a,
                "sim_out_b": sim_out_b,
                "style_ratio": sr,
                "style_gain": sg,
                "chroma_a": ca,
                "chroma_b": cb,
                "content_pres": cp,
                "label": label,
            })

            print(f"  {ns:>4.1f}  {prompt_label:>12}  {sim_out_a:>10.4f}  {sim_out_b:>10.4f}  "
                  f"{sr:>11.4f}  {sg:>+10.4f}  {ca:>13.4f}  "
                  f"{cb:>13.4f}  {cp:>12.4f}  {label:>12}")

    # ─── Section 4: 総合判定レポート ─────────────────────────────────
    section("Section 4: 総合判定レポート")

    # C1: クロスプロンプト条件
    c1 = sim_ab < 0.5

    # C2: 差分スタイル語
    c2 = len(diff_only) > 0

    # C3: 少なくとも1つのパラメータ帯で style_ratio > 1.0
    ideal_rows = [r for r in results_table if r["label"] == "IDEAL"]
    style_only_rows = [r for r in results_table if r["label"] == "STYLE_ONLY"]
    c3_any_style  = any(r["style_ratio"] > 1.0 for r in results_table)
    c3_ideal      = len(ideal_rows) > 0

    # 差分語と標準語で結果を比較
    std_best = max(
        (r for r in results_table if r["prompt_label"] == "standard"),
        key=lambda r: r["style_ratio"],
        default=None,
    )
    diff_best = max(
        (r for r in results_table if r["prompt_label"] == "differential"),
        key=lambda r: r["style_ratio"],
        default=None,
    )

    info(f"チャレンジ1 [クロスプロンプト条件]: sim(A,B)={sim_ab:.4f}  → {'PASS' if c1 else 'FAIL'}")
    info(f"チャレンジ2 [差分スタイル語抽出]:  diff_only={diff_only}  → {'PASS' if c2 else 'FAIL'}")
    info(f"チャレンジ3a [転送効果 style_ratio > 1.0]: {'あり' if c3_any_style else 'なし'}")
    info(f"チャレンジ3b [IDEAL (style+content 両立)]: {len(ideal_rows)} 件")

    if std_best:
        info(f"\n  標準スタイル語: best ns={std_best['ns']:.1f}, "
             f"ratio={std_best['style_ratio']:.4f}, cp={std_best['content_pres']:.4f}, "
             f"label={std_best['label']}")
    if diff_best:
        info(f"  差分スタイル語: best ns={diff_best['ns']:.1f}, "
             f"ratio={diff_best['style_ratio']:.4f}, cp={diff_best['content_pres']:.4f}, "
             f"label={diff_best['label']}")

    check("C1: クロスプロンプト条件成立", c1, f"sim(A,B)={sim_ab:.4f}")
    check("C2: B固有の差分スタイル語が存在", c2)
    check("C3a: style_ratio > 1.0 が少なくとも1点で成立", c3_any_style)
    check("C3b: IDEAL (style + content 両立) が少なくとも1点で成立", c3_ideal)

    # Pareto フロンティアの表示
    print(f"\n  ── Pareto フロンティア (style_ratio 上位5件) ──")
    top5 = sorted(results_table, key=lambda r: r["style_ratio"], reverse=True)[:5]
    for r in top5:
        print(f"     ns={r['ns']:.1f}  [{r['prompt_label']:>12}]  "
              f"style_ratio={r['style_ratio']:.4f}  "
              f"content_pres={r['content_pres']:.4f}  → {r['label']}")

    # IDEAL が存在する場合は推奨設定を表示
    if ideal_rows:
        best_ideal = max(ideal_rows, key=lambda r: r["style_ratio"])
        print(f"\n  ★ 推奨設定: noise_strength={best_ideal['ns']:.1f}, "
              f"prompt='{best_ideal['prompt_label']}'\n"
              f"     style_ratio={best_ideal['style_ratio']:.4f}, "
              f"content_pres={best_ideal['content_pres']:.4f}")
    else:
        print("\n  ⚠ IDEAL 条件 (style+content 両立) は得られなかった。")
        if style_only_rows:
            best_style = max(style_only_rows, key=lambda r: r["style_ratio"])
            print(f"     スタイルのみ: ns={best_style['ns']:.1f}, "
                  f"ratio={best_style['style_ratio']:.4f}\n"
                  f"     → guidance_scale を下げるか noise_strength を下げると\n"
                  f"       コンテンツ保持が改善する可能性があります。")

    # 各指標の診断
    if not c1:
        print("\n  C1失敗: sim(A,B) が高すぎる。")
        print("     → プロンプトをより対照的に (例: 'lullaby' vs 'heavy metal') してください。")
    if not c2:
        print("\n  C2失敗: B 固有の差分スタイル語がない。")
        print("     → pool_size を増やすか、プロンプトの差異をさらに広げてください。")
    if not c3_any_style:
        print("\n  C3a失敗: style_ratio が 1.0 を超えない。")
        print("     → noise_strength または guidance_scale を大きくして実験してください。")

    info(f"\n生成音声の保存先: {tmpdir}")


# ─────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IEC クロスプロンプト スタイル転送 検証スクリプト (改訂版)"
    )
    parser.add_argument("--integration", action="store_true",
                        help="統合テストを実行 (AudioLDM モデルが必要)")
    parser.add_argument("--model", default="audioldm-s-full",
                        help="使用する AudioLDM モデル名")
    parser.add_argument("--prompt-a", default="quiet piano melody",
                        help="Parent A のプロンプト (コンテンツ源)")
    parser.add_argument("--prompt-b", default="intense electronic music with heavy bass",
                        help="Parent B のプロンプト (スタイル源)。A と対照的なものを選ぶ")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="音声長 (秒)")
    parser.add_argument("--guidance-scale", type=float, default=2.5,
                        help="SDEdit の CFG スケール")
    args = parser.parse_args()

    run_section0()

    if args.integration:
        run_integration(args)
    else:
        section("統合テストはスキップ (--integration で有効化)")
        info("単体テストのみ完了")
        info(f"  例: python {sys.argv[0]} --integration --prompt-a 'quiet piano melody'")
        info(f"          --prompt-b 'intense electronic music with heavy bass'")

    print(f"\n{'═'*60}")
    print(f"  最終結果: PASS={PASS_COUNT}  FAIL={FAIL_COUNT}")
    print(f"{'═'*60}")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
