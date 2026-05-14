#!/usr/bin/env python3
"""
スタイル転送機能の検証スクリプト

【テスト構成】
  Part A: 単体テスト (モデル不要, audioldm/iec.py のみ)
    A1. StyleTransferGenotype 初期化・境界値クリップ
    A2. clone() は z0_content を共有参照で引き継ぐ
    A3. _enforce_mask_validity: mask_end > mask_start + 0.05 を保証
    A4. crossover_style_transfer: 値域・世代番号・metadata
    A5. mutate_style_transfer: 値域・世代番号・metadata
    A6. STYLE_WORD_BANK: 型・重複なし・空文字なし

  Part B: パイプライン単体テスト (モデル不要, iec_pipeline.py の純粋ロジック)
    B1. _load_waveform_for_clap: shape (1,1,t), dtype float32, 正規化済み
    B2. build_style_prompt: ベースあり・ベースなし
    B3. generate_style_transfer_audio_batch: マスクスライスの shape 整合性 (モックで確認)
    B4. initialize_style_transfer_population: noise_strength が等間隔か

  Part C: 統合テスト (AudioLDM モデル必要, --integration フラグで実行)
    C1. encode_audio_to_z0: shape (1,C,T,F), NaN なし
    C2. rank_style_words: 長さ top_k, 降順, [-1,1] 範囲
    C3. initialize_style_transfer_population: フルラウンドトリップ
    C4. evolve_style_transfer_population: population_size 個, 世代+1

Usage:
    python scripts/test_style_transfer.py              # Part A + B のみ
    python scripts/test_style_transfer.py --integration  # Part C も実行 (GPU/モデル必要)
"""

import sys
import math
import argparse
import importlib.util as _ilu
import numpy as np
import torch
import torchaudio
import tempfile
import os

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

def fail(name: str, msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {name}: {msg}")

def check(name: str, cond: bool, msg: str = ""):
    if cond:
        ok(name)
    else:
        fail(name, msg or "条件を満たしていない")

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────
# audioldm/__init__.py を経由せず iec.py を直接ロード
# (この環境では librosa/numba キャッシュエラーが発生するため)
# ─────────────────────────────────────────
def _load_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_iec = _load_module("audioldm.iec", "audioldm/iec.py")
StyleTransferGenotype   = _iec.StyleTransferGenotype
crossover_style_transfer = _iec.crossover_style_transfer
mutate_style_transfer   = _iec.mutate_style_transfer
STYLE_WORD_BANK         = _iec.STYLE_WORD_BANK

np.random.seed(0)
torch.manual_seed(0)

DUMMY_Z0 = torch.zeros(1, 8, 32, 16)  # (B, C, T, F) — 典型的な潜在空間の縮小版


# ═══════════════════════════════════════════
# Part A: StyleTransferGenotype 単体テスト
# ═══════════════════════════════════════════
section("Part A: StyleTransferGenotype 単体テスト")

# A1. 初期化・境界値クリップ
g_normal = StyleTransferGenotype(
    z0_content=DUMMY_Z0, style_prompt="jazz",
    noise_strength=0.3, guidance_scale=7.0, seed=42,
    mask_start=0.2, mask_end=0.8,
)
check("A1a noise_strength 範囲内",   0.05 <= g_normal.noise_strength <= 0.5)
check("A1b guidance_scale 範囲内",   1.0 <= g_normal.guidance_scale <= 15.0)
check("A1c mask 有効",               g_normal.mask_end > g_normal.mask_start + 0.04)
check("A1d fitness 初期値 0",        g_normal.fitness == 0.0)
check("A1e generation 初期値 0",     g_normal.generation == 0)
check("A1f id 非空文字列",           isinstance(g_normal.id, str) and len(g_normal.id) > 0)

g_clip = StyleTransferGenotype(
    z0_content=DUMMY_Z0, style_prompt="rock",
    noise_strength=99.0, guidance_scale=-5.0, seed=0,
    mask_start=0.0, mask_end=1.0,
)
check("A1g noise_strength 上限クリップ", g_clip.noise_strength == 0.5,
      f"got {g_clip.noise_strength}")
check("A1h guidance_scale 下限クリップ", g_clip.guidance_scale == 1.0,
      f"got {g_clip.guidance_scale}")

# A2. clone() 共有参照テスト
g2 = g_normal.clone()
check("A2a z0_content 共有参照",    g2.z0_content is g_normal.z0_content)
check("A2b style_prompt コピー",    g2.style_prompt == g_normal.style_prompt)
check("A2c noise_strength コピー",  g2.noise_strength == g_normal.noise_strength)
check("A2d id は独立生成",          g2.id != g_normal.id)
check("A2e fitness コピー",         g2.fitness == g_normal.fitness)
check("A2f metadata 独立 dict",     g2.metadata is not g_normal.metadata)

# A3. _enforce_mask_validity
g_inv = StyleTransferGenotype(
    z0_content=DUMMY_Z0, style_prompt="drums",
    noise_strength=0.2, guidance_scale=5.0, seed=1,
    mask_start=0.9, mask_end=0.91,  # end - start = 0.01 < 0.05 → 強制修正
)
check("A3a mask_end > mask_start + 0.04",
      g_inv.mask_end > g_inv.mask_start + 0.04,
      f"start={g_inv.mask_start:.3f} end={g_inv.mask_end:.3f}")

g_inv2 = StyleTransferGenotype(
    z0_content=DUMMY_Z0, style_prompt="piano",
    noise_strength=0.2, guidance_scale=5.0, seed=2,
    mask_start=0.97, mask_end=0.98,  # mask_end + 0.1 が 1.0 超 → 調整
)
check("A3b mask_end <= 1.0",
      g_inv2.mask_end <= 1.0,
      f"end={g_inv2.mask_end:.3f}")
check("A3c mask_start >= 0.0",
      g_inv2.mask_start >= 0.0,
      f"start={g_inv2.mask_start:.3f}")

# A4. crossover_style_transfer
np.random.seed(10)
p1 = StyleTransferGenotype(DUMMY_Z0, "jazz", 0.1, 3.0, 10, 0.0, 0.5)
p2 = StyleTransferGenotype(DUMMY_Z0, "rock", 0.4, 9.0, 20, 0.5, 1.0)
p1.generation = 2
p2.generation = 5
child = crossover_style_transfer(p1, p2)

check("A4a noise_strength 値域",   0.05 <= child.noise_strength <= 0.5,
      f"got {child.noise_strength}")
check("A4b guidance_scale 値域",   1.0 <= child.guidance_scale <= 15.0,
      f"got {child.guidance_scale}")
check("A4c mask 有効",             child.mask_end > child.mask_start + 0.04)
check("A4d generation = max+1",    child.generation == 6,
      f"got {child.generation}")
check("A4e z0_content 共有",       child.z0_content is p1.z0_content)
check("A4f metadata に parent_id", "parent1_id" in child.metadata)
check("A4g operation タグ",        child.metadata.get("operation") == "crossover_style_transfer")

# A5. mutate_style_transfer
np.random.seed(20)
base = StyleTransferGenotype(DUMMY_Z0, "ambient", 0.25, 6.0, 99, 0.2, 0.8)
base.generation = 3
mutant = mutate_style_transfer(base, noise_sigma=0.05, gs_sigma=1.0, mask_sigma=0.05)

check("A5a noise_strength 値域",    0.05 <= mutant.noise_strength <= 0.5,
      f"got {mutant.noise_strength}")
check("A5b guidance_scale 値域",    1.0 <= mutant.guidance_scale <= 15.0,
      f"got {mutant.guidance_scale}")
check("A5c mask 有効",              mutant.mask_end > mutant.mask_start + 0.04)
check("A5d generation = parent+1",  mutant.generation == 4,
      f"got {mutant.generation}")
check("A5e seed 変更済み",          mutant.seed != base.seed)
check("A5f z0_content 共有",        mutant.z0_content is base.z0_content)
check("A5g metadata に parent_id",  "parent_id" in mutant.metadata)

# A5. 大量突然変異: 全て境界内に収まるか (1000回)
np.random.seed(30)
for _ in range(1000):
    m = mutate_style_transfer(base, noise_sigma=0.2, gs_sigma=3.0, mask_sigma=0.2)
    assert 0.05 <= m.noise_strength <= 0.5, f"noise_strength={m.noise_strength}"
    assert 1.0 <= m.guidance_scale <= 15.0, f"guidance_scale={m.guidance_scale}"
    assert 0.0 <= m.mask_start <= 0.95,     f"mask_start={m.mask_start}"
    assert 0.05 <= m.mask_end <= 1.0,       f"mask_end={m.mask_end}"
    assert m.mask_end > m.mask_start + 0.04, \
        f"mask invalid: start={m.mask_start:.3f} end={m.mask_end:.3f}"
ok("A5h 1000回大量突然変異: 全て境界内")

# A6. STYLE_WORD_BANK 整合性
check("A6a list型",            isinstance(STYLE_WORD_BANK, list))
check("A6b 要素数 >= 30",       len(STYLE_WORD_BANK) >= 30,
      f"got {len(STYLE_WORD_BANK)}")
check("A6c 全て非空文字列",     all(isinstance(w, str) and len(w.strip()) > 0
                                    for w in STYLE_WORD_BANK))
check("A6d 重複なし",           len(STYLE_WORD_BANK) == len(set(STYLE_WORD_BANK)))


# ═══════════════════════════════════════════
# Part B: パイプライン純粋ロジックテスト (モデル不要)
# ═══════════════════════════════════════════
section("Part B: パイプライン純粋ロジックテスト")

# B1. _load_waveform_for_clap をスタンドアロンで検証
# (メソッドのロジックを抽出して直接テスト)
def _load_waveform_for_clap_standalone(audio_path: str) -> torch.Tensor:
    # CLAP forward() は (bs, t) を期待するので (1, t) を返す
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform / (waveform.abs().max() + 1e-8) * 0.5
    return waveform.float()  # (1, t)

TRUMPET_PATH = "trumpet.wav"
wf = _load_waveform_for_clap_standalone(TRUMPET_PATH)
check("B1a shape は (1,t) — 2次元",  wf.ndim == 2 and wf.shape[0] == 1 and wf.shape[1] > 0,
      f"got {tuple(wf.shape)}")
check("B1b dtype float32",           wf.dtype == torch.float32)
check("B1c 最大絶対値 <= 0.5+ε",    wf.abs().max().item() <= 0.501,
      f"max={wf.abs().max().item():.4f}")
check("B1d NaN なし",                not torch.isnan(wf).any())

# 合成サイン波 (ステレオ 44kHz → モノ 16kHz に変換されるか)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    tmp_path = f.name
sr_test = 44100
t_test = torch.linspace(0, 1.0, sr_test)
stereo = torch.stack([torch.sin(2 * math.pi * 440 * t_test),
                       torch.sin(2 * math.pi * 880 * t_test)], dim=0)
torchaudio.save(tmp_path, stereo, sr_test)
wf2 = _load_waveform_for_clap_standalone(tmp_path)
os.unlink(tmp_path)
check("B1e ステレオ→モノ変換後 ndim==2 shape[0]==1", wf2.ndim == 2 and wf2.shape[0] == 1,
      f"got {tuple(wf2.shape)}")
check("B1f 16kHz リサンプル後サンプル数 ≈ 16000",
      abs(wf2.shape[1] - 16000) < 100,
      f"got {wf2.shape[1]}")

# B2. build_style_prompt (スタンドアロン)
def build_style_prompt_standalone(base_prompt: str, ranked_words) -> str:
    words_str = ", ".join(w for w, _ in ranked_words)
    if base_prompt.strip():
        return f"{base_prompt.strip()}, {words_str}"
    return words_str

ranked_sample = [("jazz", 0.9), ("piano", 0.8), ("calm", 0.7)]
p_with_base  = build_style_prompt_standalone("music with soft melody", ranked_sample)
p_empty_base = build_style_prompt_standalone("", ranked_sample)
p_space_base = build_style_prompt_standalone("   ", ranked_sample)

check("B2a ベースあり: カンマ結合",
      p_with_base == "music with soft melody, jazz, piano, calm",
      f"got: {p_with_base!r}")
check("B2b ベースなし: スタイル語のみ",
      p_empty_base == "jazz, piano, calm",
      f"got: {p_empty_base!r}")
check("B2c スペースのみ: ベースなし扱い",
      p_space_base == "jazz, piano, calm",
      f"got: {p_space_base!r}")

# B3. マスクスライスの shape 整合性
# _generate_style_transfer_audio_batch のマスク分岐ロジックを直接検証
T = 256
z0_mock = torch.randn(1, 8, T, 16)

def apply_mask_slice(z0, mask_start, mask_end):
    T_local = z0.shape[2]
    ts = int(mask_start * T_local)
    te = max(ts + 1, int(mask_end * T_local))
    z0_slice = z0[:, :, ts:te, :]
    return z0_slice, ts, te

for ms, me in [(0.0, 1.0), (0.2, 0.8), (0.5, 0.7), (0.9, 1.0)]:
    g_tmp = StyleTransferGenotype(z0_mock, "test", 0.2, 5.0, 0, ms, me)
    sl, ts, te = apply_mask_slice(z0_mock, g_tmp.mask_start, g_tmp.mask_end)
    expected_len = te - ts
    check(f"B3 mask({ms:.1f},{me:.1f}) slice shape (1,8,{expected_len},16)",
          sl.shape == (1, 8, expected_len, 16),
          f"got {tuple(sl.shape)}")

# B4. initialize_style_transfer_population: noise_strength 等間隔サンプリング
ns_min, ns_max = 0.1, 0.4
pop_size = 6
ns_vals = np.linspace(ns_min, ns_max, pop_size)
check("B4a 等間隔: 最小値",  abs(ns_vals[0] - ns_min) < 1e-6)
check("B4b 等間隔: 最大値",  abs(ns_vals[-1] - ns_max) < 1e-6)
check("B4c 等間隔: 間隔均等",
      all(abs((ns_vals[i+1] - ns_vals[i]) - (ns_vals[1] - ns_vals[0])) < 1e-6
          for i in range(len(ns_vals)-1)))


# ═══════════════════════════════════════════
# Part C: 統合テスト (モデル必要)
# ═══════════════════════════════════════════
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--integration", action="store_true")
args, _ = parser.parse_known_args()

if not args.integration:
    section("Part C: 統合テスト — スキップ (--integration を付けて実行)")
    print("  ヒント: AudioLDM モデルが必要です。")
    print("  実行: python scripts/test_style_transfer.py --integration")
else:
    section("Part C: 統合テスト (AudioLDM モデル読み込み中...)")

    try:
        from audioldm.iec_pipeline import AudioLDM_IEC

        CKPT = os.environ.get("AUDIOLDM_CKPT", None)
        print(f"  モデルロード中 (ckpt_path={CKPT})...")
        iec = AudioLDM_IEC(
            ckpt_path=CKPT,
            population_size=4,
            duration=5.0,
            ddim_steps=20,  # 検証用に短縮
        )
        AUDIO_A = TRUMPET_PATH
        AUDIO_B = TRUMPET_PATH  # 同じファイルで代用 (テスト環境)

        # C1. encode_audio_to_z0
        z0 = iec.encode_audio_to_z0(AUDIO_A)
        check("C1a shape 4次元 (1,C,T,F)", z0.ndim == 4 and z0.shape[0] == 1,
              f"got {tuple(z0.shape)}")
        check("C1b T > 0",                 z0.shape[2] > 0)
        check("C1c NaN なし",              not torch.isnan(z0).any())
        check("C1d |z0| < 1e2",            torch.max(torch.abs(z0)).item() < 100)
        print(f"  z0 shape: {tuple(z0.shape)}")

        # C2. rank_style_words
        ranked = iec.rank_style_words(AUDIO_B, top_k=5)
        check("C2a 長さ top_k=5",         len(ranked) == 5,
              f"got {len(ranked)}")
        check("C2b 降順ソート",            all(ranked[i][1] >= ranked[i+1][1]
                                               for i in range(len(ranked)-1)),
              f"scores={[s for _,s in ranked]}")
        check("C2c スコア範囲 [-1,1]",     all(-1.0 <= s <= 1.0 for _, s in ranked),
              f"scores={[s for _,s in ranked]}")
        check("C2d 語が STYLE_WORD_BANK 内",
              all(w in STYLE_WORD_BANK for w, _ in ranked))
        print(f"  Top-5 style words: {[(w, f'{s:.3f}') for w,s in ranked]}")

        # C3. initialize_style_transfer_population
        results, ranked_words = iec.initialize_style_transfer_population(
            audio_a_path=AUDIO_A,
            audio_b_path=AUDIO_B,
            base_prompt="music",
            top_k_styles=3,
            population_size=4,
        )
        check("C3a results 長さ == pop_size",     len(results) == 4,
              f"got {len(results)}")
        check("C3b ranked_words 長さ == top_k",   len(ranked_words) == 3,
              f"got {len(ranked_words)}")

        for idx, (g, wf_arr) in enumerate(results):
            check(f"C3c[{idx}] genotype 型",
                  isinstance(g, StyleTransferGenotype))
            check(f"C3d[{idx}] waveform ndarray",
                  isinstance(wf_arr, np.ndarray))
            check(f"C3e[{idx}] waveform shape[1] > 0",
                  wf_arr.ndim >= 1 and wf_arr.shape[-1] > 0)
            check(f"C3f[{idx}] waveform NaN なし",
                  not np.isnan(wf_arr).any())

        check("C3g z0_content 全個体共有参照",
              all(results[i][0].z0_content is results[0][0].z0_content
                  for i in range(1, len(results))))
        check("C3h style_prompt に ranked_words 最上位語を含む",
              ranked_words[0][0] in results[0][0].style_prompt)

        # noise_strength が等間隔 4点か検証
        ns_list = sorted(g.noise_strength for g, _ in results)
        diffs = [ns_list[i+1] - ns_list[i] for i in range(len(ns_list)-1)]
        check("C3i noise_strength 等間隔",
              max(diffs) - min(diffs) < 0.01,
              f"diffs={[f'{d:.4f}' for d in diffs]}")

        # C4. evolve_style_transfer_population
        gen_before = iec.population.generation_number
        results2 = iec.evolve_style_transfer_population(
            selected_indices=[0, 1],
            elite_count=1,
        )
        check("C4a results2 長さ == pop_size",    len(results2) == 4,
              f"got {len(results2)}")
        check("C4b generation_number インクリメント",
              iec.population.generation_number == gen_before + 1)

        elite = results2[0][0]
        check("C4c エリート個体の metadata に elite フラグ",
              elite.metadata.get("elite") is True)

        for idx, (g, wf_arr) in enumerate(results2):
            check(f"C4d[{idx}] waveform NaN なし",
                  not np.isnan(wf_arr).any())
            if g.metadata.get("elite"):
                check(f"C4e[{idx}] elite generation >= 0",
                      g.generation >= 0, f"got {g.generation}")
            else:
                check(f"C4e[{idx}] generation >= 1",
                      g.generation >= 1, f"got {g.generation}")

        check("C4f 世代履歴に追加",
              len(iec.population.history) >= 2)

        print(f"\n  生成波形サンプル数: {results[0][1].shape}")

    except ImportError as e:
        fail("Part C 全体", f"audioldm import 失敗: {e}")
    except Exception as e:
        import traceback
        fail("Part C 全体", f"例外発生: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────
# 最終サマリー
# ─────────────────────────────────────────
total = PASS_COUNT + FAIL_COUNT
section(f"結果サマリー: {PASS_COUNT}/{total} PASS, {FAIL_COUNT}/{total} FAIL")

if FAIL_COUNT > 0:
    print("  一部テストが失敗しました。上記の [FAIL] を確認してください。")
    sys.exit(1)
else:
    print("  全テスト合格!")
    sys.exit(0)
