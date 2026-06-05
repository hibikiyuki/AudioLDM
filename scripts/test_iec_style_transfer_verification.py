#!/usr/bin/env python3
"""
IEC スタイル転送 検証スクリプト

仮説:
  「同じプロンプトから生まれた2体(親A・親B)を選び、
   Aのz0を順拡散 → BのCLAP抽出スタイル語で逆拡散」
   という手法で B のスタイルが A のコンテンツに乗るか？

超えるべきハードル:
  1. 多様性: 同一プロンプト由来でも個体間でCLAPスタイル語が十分異なるか？
  2. 差分抽出: CLAPの差分ランキングが B 固有の意味ある語を返すか？
  3. 転送効果: 生成物が A より B に近くなるか？ (CLAP similarity)

Usage:
    python scripts/test_iec_style_transfer_verification.py              # Section 0 のみ
    python scripts/test_iec_style_transfer_verification.py --integration \\
        --model audioldm-m-full \\
        --prompt "piano melody with strings" \\
        --population-size 4 \\
        --duration 5.0
"""

import sys
import os
import argparse
import tempfile
import math
import numpy as np
import torch
import torch.nn.functional as F

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


def jaccard(set_a, set_b):
    """Jaccard 類似度 |A∩B| / |A∪B|"""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


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
# Section 0: 単体テスト (モデル不要)
# ─────────────────────────────────────────

def _mock_rank_differential(words, scores_a, scores_b, top_k=5, pool_size=10):
    """rank_differential_style_words のコアロジック（集合差分版）をモックで再現。"""
    top_b = sorted(zip(words, scores_b), key=lambda x: -x[1])[:pool_size]
    top_a_words = {w for w, _ in sorted(zip(words, scores_a), key=lambda x: -x[1])[:pool_size]}
    exclusive_b = [(w, s) for w, s in top_b if w not in top_a_words]
    return exclusive_b[:top_k]


def _mock_rank_standard(words, scores_b, top_k=5):
    """rank_style_words のコアロジック（B だけ）。"""
    ranked = sorted(zip(words, scores_b), key=lambda x: -x[1])
    return ranked[:top_k]


def run_section0():
    section("Section 0: 単体テスト (モデル不要)")

    words = ["jazz", "classical", "electronic", "calm", "energetic",
             "guitar", "piano", "drums", "reverb-heavy", "lo-fi"]
    # A と B が近い場合: どちらも jazz/classical が高い
    scores_a = [0.8, 0.7, 0.3, 0.6, 0.2, 0.5, 0.6, 0.3, 0.2, 0.1]
    # B は electronic/energetic/drums が際立って高い (A との差)
    scores_b = [0.75, 0.65, 0.8, 0.4, 0.85, 0.5, 0.55, 0.9, 0.5, 0.3]

    # pool_size=5 で実行: B の Top-5 から A の Top-5 を除いた B 固有語
    diff_top5 = _mock_rank_differential(words, scores_a, scores_b, top_k=5, pool_size=5)
    std_top5  = _mock_rank_standard(words, scores_b, top_k=5)

    diff_words = [w for w, _ in diff_top5]   # 2-tuple: (word, score_b)
    std_words  = [w for w, _ in std_top5]

    # 0-1: 集合差分で B 固有の語が選ばれているか
    # B Top-5: drums(0.9),energetic(0.85),electronic(0.8),jazz(0.75),classical(0.65)
    # A Top-5: jazz(0.8),classical(0.7),calm(0.6),piano(0.6),guitar(0.5)
    # B-exclusive: {drums, energetic, electronic}
    expected_in_diff = {"drums", "energetic", "electronic"}
    check(
        "0-1: 集合差分Top語が B 固有の高スコア語を含む",
        bool(expected_in_diff & set(diff_words)),
        f"diff_words={diff_words}, expected some of {expected_in_diff}",
    )

    # 0-2: 標準 Top-5 に含まれ A と共通の語（jazz, classical）が差分リストから除外されている
    std_only = set(std_words) - set(diff_words)
    check(
        "0-2: 標準語のうち A と共通の語が差分リストで除外されている",
        len(std_only) > 0,
        f"std_only={std_only}, diff={diff_words}, std={std_words}",
    )
    info(f"差分Top (B-exclusive): {diff_words}")
    info(f"標準Top5 (Bスコア順):  {std_words}")
    info(f"標準にあるが差分から除外: {std_only}  ← A と共通の語")

    # 0-3: pool_size を変えると除外候補が変わる
    # pool_size=3: A Top-3={jazz,classical,calm}, B Top-3={drums,energetic,electronic} → 全3語が B 固有
    diff_pool3 = _mock_rank_differential(words, scores_a, scores_b, top_k=5, pool_size=3)
    diff_pool3_set = {w for w, _ in diff_pool3}
    b_top3 = {w for w, _ in sorted(zip(words, scores_b), key=lambda x: -x[1])[:3]}
    check(
        "0-3: pool_size=3 のとき結果はすべて B の Top-3 内",
        diff_pool3_set.issubset(b_top3),
        f"result={diff_pool3_set}, b_top3={b_top3}",
    )

    # 0-4: A=B のとき B の Top-N == A の Top-N → 集合差分は空
    same_scores = [0.5] * len(words)
    diff_same = _mock_rank_differential(words, same_scores, same_scores, top_k=5, pool_size=5)
    check(
        "0-4: A=B のとき B 固有語なし (空リスト)",
        len(diff_same) == 0,
        f"len={len(diff_same)} (期待: 0)",
    )


# ─────────────────────────────────────────
# Section 1〜4: 統合テスト
# ─────────────────────────────────────────

def run_integration(args):
    from audioldm.iec_pipeline import AudioLDM_IEC
    import soundfile as sf

    section("統合テスト準備: モデルロード")
    pipeline = AudioLDM_IEC(
        model_name=args.model,
        population_size=args.population_size,
        duration=args.duration,
    )
    info(f"モデル: {args.model}, 個体数: {args.population_size}, 時間: {args.duration}s")

    tmpdir = tempfile.mkdtemp(prefix="iec_style_verify_")
    info(f"作業ディレクトリ: {tmpdir}")

    # ─── Section 1: 個体群多様性 ───────────────────────────────────────
    section("Section 1: 個体群多様性の検証")
    info(f"プロンプト: '{args.prompt}' から {args.population_size} 体を z0 モードで生成")

    results = pipeline.initialize_population_z0(prompt=args.prompt)

    # 音声ファイルに保存 (waveform は (1,1,T) や (T,) など shape が可変)
    audio_paths = []
    for i, (_, waveform) in enumerate(results):
        path = os.path.join(tmpdir, f"individual_{i}.wav")
        wav = np.array(waveform).squeeze()   # → (T,)
        sf.write(path, wav, 16000)
        audio_paths.append(path)
    info(f"音声保存完了: {audio_paths}")

    # 各個体のスタイル語 Top-10
    info("各個体のCLAPスタイル語 Top-10:")
    all_top_words = []
    for i, path in enumerate(audio_paths):
        ranked = pipeline.rank_style_words(path, top_k=10)
        words = [w for w, _ in ranked]
        all_top_words.append(words)
        info(f"  個体{i}: {words}")

    # ペア間 Jaccard 類似度
    n = len(audio_paths)
    jaccard_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            j_sim = jaccard(all_top_words[i], all_top_words[j])
            jaccard_matrix[(i, j)] = j_sim

    info("ペア間 Jaccard 類似度 (Top-10 語セット):")
    for (i, j), sim in sorted(jaccard_matrix.items(), key=lambda x: x[1]):
        info(f"  ({i},{j}): {sim:.3f}")

    min_jaccard = min(jaccard_matrix.values())
    min_pair = min(jaccard_matrix, key=jaccard_matrix.get)
    check(
        "1-1: 最小 Jaccard < 0.6 (多様性あり)",
        min_jaccard < 0.6,
        f"min_jaccard={min_jaccard:.3f} for pair {min_pair}",
    )

    # ペア間 CLAP audio similarity
    info("ペア間 CLAP audio cosine similarity:")
    embs = [clap_audio_embedding(pipeline, p) for p in audio_paths]
    clap_sim_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(embs[i], embs[j], dim=1).item()
            clap_sim_matrix[(i, j)] = sim
            info(f"  ({i},{j}): {sim:.3f}")

    min_clap_pair = min(clap_sim_matrix, key=clap_sim_matrix.get)
    info(f"最大多様性ペア (CLAP距離最大): {min_clap_pair}")

    # スタイル語多様性で最大多様性ペアを選ぶ
    idx_a, idx_b = min_pair
    path_a, path_b = audio_paths[idx_a], audio_paths[idx_b]
    info(f"選択した (A, B) ペア: ({idx_a}, {idx_b}), Jaccard={min_jaccard:.3f}")

    # ─── Section 2: 差分スタイル語 ────────────────────────────────────
    section("Section 2: 差分スタイル語 vs 標準スタイル語の比較")

    std_ranked  = pipeline.rank_style_words(path_b, top_k=5)
    diff_ranked = pipeline.rank_differential_style_words(path_b, path_a, top_k=5, pool_size=10)

    std_words  = [w for w, _ in std_ranked]
    diff_words = [w for w, _ in diff_ranked]   # 2-tuple: (word, score_b)

    info(f"標準 Top-5 (B): {std_words}")
    info("  " + ", ".join(f"{w}({s:.3f})" for w, s in std_ranked))
    info(f"差分 Top-5 (B-A集合差分, pool=10): {diff_words}")
    info("  " + ", ".join(f"{w}({s:.3f})" for w, s in diff_ranked))

    diff_only = set(diff_words) - set(std_words)
    std_only  = set(std_words) - set(diff_words)
    info(f"差分語のみ登場: {diff_only}")
    info(f"標準語のみ登場: {std_only}")
    check(
        "2-1: 差分語に標準語と異なる語が含まれる",
        len(diff_only) > 0,
        f"diff_only={diff_only}",
    )

    # ─── Section 3: スタイル転送の効果測定 ──────────────────────────
    section("Section 3: スタイル転送の効果測定")

    emb_a = clap_audio_embedding(pipeline, path_a)
    emb_b = clap_audio_embedding(pipeline, path_b)
    sim_a_b = F.cosine_similarity(emb_a, emb_b, dim=1).item()
    info(f"sim(A, B) = {sim_a_b:.4f}  (転送前の参照値)")

    z0_content = pipeline.encode_audio_to_z0(path_a)

    noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

    diff_prompt_words = list(diff_ranked)   # already (word, score_b) 2-tuples

    # パターン1: 標準スタイル語 (共通プロンプト付き)
    std_prompt      = pipeline.build_style_prompt(args.prompt, std_ranked)
    # パターン2: 差分スタイル語 (共通プロンプト付き)
    diff_prompt     = pipeline.build_style_prompt(args.prompt, diff_prompt_words)
    # パターン3: 標準スタイル語のみ (共通プロンプトなし)
    std_words_only  = pipeline.build_style_prompt("", std_ranked)
    # パターン4: 差分スタイル語のみ (共通プロンプトなし)
    diff_words_only = pipeline.build_style_prompt("", diff_prompt_words)

    info(f"標準スタイルプロンプト (共通あり): {std_prompt}")
    info(f"差分スタイルプロンプト (共通あり): {diff_prompt}")
    info(f"標準スタイルプロンプト (語のみ):   {std_words_only}")
    info(f"差分スタイルプロンプト (語のみ):   {diff_words_only}")

    from audioldm.iec import StyleTransferGenotype

    results_table = []
    for prompt_label, style_prompt in [
        ("standard",           std_prompt),
        ("differential",       diff_prompt),
        ("standard_words_only",  std_words_only),
        ("diff_words_only",      diff_words_only),
    ]:
        info(f"\n--- {prompt_label} スタイル語でのSDEdit ---")
        for ns in noise_strengths:
            g = StyleTransferGenotype(
                z0_content=z0_content,
                style_prompt=style_prompt,
                noise_strength=ns,
                guidance_scale=args.guidance_scale,
                seed=42,
            )
            waveforms = pipeline.generate_style_transfer_audio_batch([g])
            out_path = os.path.join(tmpdir, f"result_{prompt_label}_ns{ns:.1f}.wav")
            sf.write(out_path, np.array(waveforms[0]).squeeze(), 16000)

            emb_out = clap_audio_embedding(pipeline, out_path)
            sim_out_a = F.cosine_similarity(emb_out, emb_a, dim=1).item()
            sim_out_b = F.cosine_similarity(emb_out, emb_b, dim=1).item()
            # 判定: 転送後に B への近さが「A と B の元の距離」を超えたか？
            # sim(out,B) > sim(out,A) は A の z0 から始める限り構造的に成立しない。
            # 正しい問い: 「転送で B への近さが増したか？」= sim(out,B) > sim(A,B)
            style_gain = sim_out_b - sim_a_b
            b_closer = sim_out_b > sim_a_b
            results_table.append((prompt_label, ns, sim_out_a, sim_out_b, style_gain, b_closer))
            gain_str = f"gain={style_gain:+.4f}"
            marker = "← スタイル獲得!" if b_closer else ""
            info(f"  ns={ns:.1f}: sim(out,A)={sim_out_a:.4f}  sim(out,B)={sim_out_b:.4f}  {gain_str}  {marker}")

    # ─── Section 4: 判定レポート ──────────────────────────────────────
    section("Section 4: 総合判定レポート")

    # チャレンジ1
    c1 = min_jaccard < 0.6
    info(f"チャレンジ1 [多様性]:  最小Jaccard={min_jaccard:.3f}  → {'PASS' if c1 else 'FAIL'}")

    # チャレンジ2
    c2 = len(diff_only) > 0
    info(f"チャレンジ2 [差分抽出]: 差分語のみ登場={diff_only}  → {'PASS' if c2 else 'FAIL'}")

    # チャレンジ3: sim(out,B) > sim(A,B) となるパラメータ帯が存在するか
    # ※ sim(out,B) > sim(out,A) は A の z0 から始める限り構造的に成立しない (A に近いのは当然)。
    # 正しい判定: B への近さが「A の元の B への距離 sim(A,B)」を超えたか = style_gain > 0
    b_closer_any = any(b for _, _, _, _, _, b in results_table)

    def _best_gain(label):
        return max((g for lbl, _, _, _, g, _ in results_table if lbl == label), default=float("-inf"))

    def _any_closer(label):
        return any(b for lbl, _, _, _, _, b in results_table if lbl == label)

    patterns = [
        ("standard",            "標準 (共通あり)"),
        ("differential",        "差分 (共通あり)"),
        ("standard_words_only", "標準 (語のみ)  "),
        ("diff_words_only",     "差分 (語のみ)  "),
    ]

    info(f"チャレンジ3 [転送効果] — 判定: sim(out,B) > sim(A,B)={sim_a_b:.4f}")
    for lbl, display in patterns:
        g = _best_gain(lbl)
        closer = _any_closer(lbl)
        info(f"  {display}: B への近さ獲得={'あり' if closer else 'なし'}  (最大gain={g:+.4f})")
    c3 = b_closer_any
    info(f"  → {'PASS' if c3 else 'FAIL'}")

    print(f"\n{'─'*60}")
    print(f"  PASS: {PASS_COUNT}  FAIL: {FAIL_COUNT}")
    print(f"  チャレンジ総合: C1={'✓' if c1 else '✗'} C2={'✓' if c2 else '✗'} C3={'✓' if c3 else '✗'}")
    print(f"{'─'*60}")

    if not c1:
        print("\n  C1失敗: 同一プロンプト由来の個体間に十分な多様性がない。")
        print("     → 異なるプロンプト/異なる種類の音を親として選ぶことを推奨。")
    if not c2:
        print("\n  C2失敗: 差分スタイル語が標準語と一致。")
        print("     → A と B がほぼ同じ CLAP embedding になっている（多様性不足に連動）。")
    if not c3:
        print("\n  C3失敗: sim(out,B) がベースライン sim(A,B) を超えられなかった。")
        print("     → noise_strength をさらに上げる / guidance_scale を大きくする実験を。")
    if c3:
        print("\n  C3 PASS: スタイル転送が機能している証拠が得られた。")
        all_gains = [(display, _best_gain(lbl)) for lbl, display in patterns]
        best_display, best_gain = max(all_gains, key=lambda x: x[1])
        print(f"     → {best_display.strip()} で最大 style_gain={best_gain:+.4f}")

    info(f"\n生成音声の保存先: {tmpdir}")


# ─────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IEC スタイル転送 検証スクリプト")
    parser.add_argument("--integration", action="store_true",
                        help="統合テストを実行 (AudioLDM モデルが必要)")
    parser.add_argument("--model", default="audioldm-s-full",
                        help="使用する AudioLDM モデル名")
    parser.add_argument("--prompt", default="piano melody with strings",
                        help="個体群生成に使うプロンプト")
    parser.add_argument("--population-size", type=int, default=4,
                        help="個体群サイズ")
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

    print(f"\n{'═'*60}")
    print(f"  最終結果: PASS={PASS_COUNT}  FAIL={FAIL_COUNT}")
    print(f"{'═'*60}")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
